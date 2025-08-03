use wasm_bindgen::prelude::*;
use candle_core::{Tensor, Device, DType, Error as CandleError, Module, Var};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::Read;

// LoRA configuration from adapter_config.json
#[derive(Deserialize)]
struct LoRAConfig {
    r: usize,  // LoRA rank
    task_type: String,
    target_modules: Vec<String>,
    lora_alpha: f32,
    lora_dropout: f32,
    modules_to_save: Option<Vec<String>>,
}

// LoRA checkpoint structure
#[derive(Serialize, Deserialize)]
struct LoRACheckpoint {
    a: Vec<f32>,
    b: Vec<f32>,
    a_shape: (usize, usize),
    b_shape: (usize, usize),
}

// Load adapter_config.json
fn load_lora_config(path: &str) -> Result<LoRAConfig, CandleError> {
    let mut file = File::open(path).map_err(|e| CandleError::Io(e))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).map_err(|e| CandleError::Io(e))?;
    serde_json::from_str(&contents).map_err(|e| CandleError::Msg(e.to_string()))
}

// Load Safetensors checkpoint
fn load_safetensors_checkpoint(path: &str, rows: usize, cols: usize, lora_rank: usize) -> Result<LoRACheckpoint, CandleError> {
    let data = std::fs::read(path).map_err(|e| CandleError::Io(e))?;
    let safetensors = SafeTensors::deserialize(&data).map_err(|e| CandleError::Msg(e.to_string()))?;

    // Assume LoRA weights are stored as "lora_A" and "lora_B" (adjust based on actual naming)
    let a_tensor = safetensors.tensor("lora_A").map_err(|e| CandleError::Msg(e.to_string()))?;
    let b_tensor = safetensors.tensor("lora_B").map_err(|e| CandleError::Msg(e.to_string()))?;

    let a_data = a_tensor.data().to_vec();
    let b_data = b_tensor.data().to_vec();

    Ok(LoRACheckpoint {
        a: a_data,
        b: b_data,
        a_shape: (rows, lora_rank),
        b_shape: (lora_rank, cols),
    })
}

// Straight-through estimator for quantization
fn straight_through_estimator(tensor: &Tensor, quantized: &Tensor) -> Result<Tensor, CandleError> {
    tensor + (quantized - tensor).detach()?
}

// Helper function for GPTQ quantization (unchanged)
fn gptq_quantize(
    tensor: &Tensor,
    bit_depth: u8,
    block_size: usize,
    symmetric: bool
) -> Result<Tensor, CandleError> {
    let dtype = match bit_depth {
        4 => DType::QInt4,
        8 => DType::QInt8,
        _ => return Err(CandleError::UnsupportedDType),
    };

    let shape = tensor.shape();
    let block_count = (shape[0] + block_size - 1) / block_size;
    let mut quantized = Vec::new();

    for i in 0..block_count {
        let start = i * block_size;
        let end = std::cmp::min(start + block_size, shape[0]);
        let block = tensor.narrow(0, start, end - start)?;
        
        let scale = if symmetric {
            block.abs()?.max(0)?.to_scalar::<f32>()? / ((1 << (bit_depth - 1)) - 1) as f32
        } else {
            let min = block.min(0)?.to_scalar::<f32>()?;
            let max = block.max(0)?.to_scalar::<f32>()?;
            (max - min) / ((1 << bit_depth) - 1) as f32
        };

        let quantized_block = if symmetric {
            (block / scale)?.to_dtype(dtype)?
        } else {
            let min = block.min(0)?.to_scalar::<f32>()?;
            ((block - min) / scale)?.to_dtype(dtype)?
        };
        quantized.extend_from_slice(&quantized_block.to_vec1::<f32>()?);
    }

    Tensor::from_vec(quantized, shape, &Device::Cpu)
}

// Helper function for QLoRA quantization with QAT
fn qlora_quantize(
    tensor: &Tensor,
    bit_depth: u8,
    block_size: usize,
    symmetric: bool,
    lora_rank: usize,
    lora_checkpoint: &str,  // Path to adapter_model.safetensors
    qat_enabled: bool,
    learning_rate: f32
) -> Result<Tensor, CandleError> {
    let dtype = match bit_depth {
        4 => DType::QInt4,
        8 => DType::QInt8,
        _ => return Err(CandleError::UnsupportedDType),
    };

    let shape = tensor.shape();
    let [rows, cols] = shape.as_slice() else {
        return Err(CandleError::ShapeMismatch(format!("Expected 2D tensor, got {:?}", shape)));
    };

    // Load LoRA config from adapter_config.json
    let config_path = lora_checkpoint.replace("adapter_model.safetensors", "adapter_config.json");
    let lora_config = load_lora_config(&config_path)?;
    if lora_config.r != lora_rank {
        return Err(CandleError::Msg(format!(
            "LoRA rank mismatch: config r={}, provided r={}",
            lora_config.r, lora_rank
        )));
    }

    // Load pretrained LoRA matrices from Safetensors
    let checkpoint = load_safetensors_checkpoint(lora_checkpoint, *rows, *cols, lora_rank)?;
    if checkpoint.a_shape != (*rows, lora_rank) || checkpoint.b_shape != (lora_rank, *cols) {
        return Err(CandleError::ShapeMismatch(format!(
            "LoRA checkpoint shapes mismatch: A={:?}, B={:?}, expected ({}, {}), ({}, {})",
            checkpoint.a_shape, checkpoint.b_shape, rows, lora_rank, lora_rank, cols
        )));
    }

    // Create trainable variables for A and B
    let a_var = Var::from_vec(checkpoint.a, checkpoint.a_shape, &Device::Cpu)?;
    let b_var = Var::from_vec(checkpoint.b, checkpoint.b_shape, &Device::Cpu)?;

    // Compute LoRA update: ΔW = A * B
    let a = a_var.as_tensor();
    let b = b_var.as_tensor();
    let delta_w = a.matmul(&b)?;

    // Apply LoRA update: W' = W + ΔW
    let updated_tensor = tensor.add(&delta_w)?;

    // Quantize block-wise
    let block_count = (rows + block_size - 1) / block_size;
    let mut quantized = Vec::new();

    for i in 0..block_count {
        let start = i * block_size;
        let end = std::cmp::min(start + block_size, *rows);
        let block = updated_tensor.narrow(0, start, end - start)?;

        let scale = if symmetric {
            block.abs()?.max(0)?.to_scalar::<f32>()? / ((1 << (bit_depth - 1)) - 1) as f32
        } else {
            let min = block.min(0)?.to_scalar::<f32>()?;
            let max = block.max(0)?.to_scalar::<f32>()?;
            (max - min) / ((1 << bit_depth) - 1) as f32
        };

        let quantized_block = if symmetric {
            (block / scale)?.to_dtype(dtype)?
        } else {
            let min = block.min(0)?.to_scalar::<f32>()?;
            ((block - min) / scale)?.to_dtype(dtype)?
        };

        if qat_enabled {
            // Apply straight-through estimator for QAT
            let q_block = straight_through_estimator(&block, &quantized_block)?;
            quantized.extend_from_slice(&q_block.to_vec1::<f32>()?);

            // Mock gradient update (simplified for WebAssembly)
            let loss = q_block.sqr()?.mean_all()?; // Mock loss
            let grad = loss.backward()?;
            if let Some(a_grad) = grad.get(&a_var) {
                let update = a_grad.mul(&learning_rate)?;
                a_var.sub_assign(&update)?;
            }
            if let Some(b_grad) = grad.get(&b_var) {
                let update = b_grad.mul(&learning_rate)?;
                b_var.sub_assign(&update)?;
            }
        } else {
            quantized.extend_from_slice(&quantized_block.to_vec1::<f32>()?);
        }
    }

    // Serialize updated LoRA matrices back to Safetensors (for QAT persistence)
    let updated_checkpoint = LoRACheckpoint {
        a: a_var.as_tensor().to_vec1::<f32>()?,
        b: b_var.as_tensor().to_vec1::<f32>()?,
        a_shape: checkpoint.a_shape,
        b_shape: checkpoint.b_shape,
    };
    let mut safetensors_data = Vec::new();
    safetensors::serialize(
        &[
            ("lora_A", updated_checkpoint.a.as_slice(), updated_checkpoint.a_shape),
            ("lora_B", updated_checkpoint.b.as_slice(), updated_checkpoint.b_shape),
        ],
        &mut safetensors_data,
    )
    .map_err(|e| CandleError::Msg(e.to_string()))?;
    // Note: Writing to file in WebAssembly requires WASI; mock for now
    // std::fs::write(lora_checkpoint, safetensors_data).map_err(|e| CandleError::Io(e))?;

    Tensor::from_vec(quantized, shape, &Device::Cpu)
}

#[wasm_bindgen]
pub fn quantize_batch(
    weights: Vec<f32>,
    batch_size: usize,
    bit_depth: u8,
    technique: &str,
    block_size: usize,
    mode: &str,
    lora_rank: usize,
    lora_checkpoint: &str,  // Path to adapter_model.safetensors
    qat_enabled: bool,
    learning_rate: f32
) -> Result<Vec<f32>, JsValue> {
    let device = Device::Cpu;
    let symmetric = mode == "symmetric";

    // Validate inputs
    if weights.len() % batch_size != 0 {
        return Err(JsValue::from_str("Weights length must be divisible by batch_size"));
    }
    let weights_per_batch = weights.len() / batch_size;

    let mut result = Vec::new();

    for i in 0..batch_size {
        let start = i * weights_per_batch;
        let end = start + weights_per_batch;
        let batch_weights = weights[start..end].to_vec();
        let rows = (batch_weights.len() as f64).sqrt() as usize;
        let cols = batch_weights.len() / rows;
        let tensor = Tensor::from_vec(batch_weights, (rows, cols), &device)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let quantized = match technique {
            "gptq" => gptq_quantize(&tensor, bit_depth, block_size, symmetric),
            "qlora" => qlora_quantize(&tensor, bit_depth, block_size, symmetric, lora_rank, lora_checkpoint, qat_enabled, learning_rate),
            "awq" => gptq_quantize(&tensor, bit_depth, block_size, symmetric),
            "qat" => qlora_quantize(&tensor, bit_depth, block_size, symmetric, lora_rank, lora_checkpoint, true, learning_rate),
            _ => return Err(JsValue::from_str(&format!("Unsupported technique: {}", technique))),
        }
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        result.extend_from_slice(&quantized.to_vec1::<f32>().map_err(|e| JsValue::from_str(&e.to_string()))?);
    }

    Ok(result)
}

#[wasm_bindgen]
pub fn free_memory() {
    // No-op; candle manages memory internally
}