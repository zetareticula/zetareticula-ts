import { BitDepth, QuantizationConfig, QuantizationTechnique } from './types';
import { Expert } from '../moe/types';

// Extended quantization config with LoRA and QAT parameters
export interface ExtendedQuantizationConfig extends QuantizationConfig {
  blockSize?: number;
  mode: 'symmetric' | 'asymmetric';
  loraRank?: number;
  loraCheckpoint?: string; // Path to adapter_model.safetensors
  qatEnabled?: boolean;
  learningRate?: number;
}

// Type definition for the WebAssembly module
interface QuantizerWasmModule {
  quantize_batch: (
    weights: Float32Array,
    batch_size: number,
    bit_depth: number,
    technique: string,
    block_size: number,
    mode: string,
    lora_rank: number,
    lora_checkpoint: string,
    qat_enabled: boolean,
    learning_rate: number
  ) => Float32Array;
  free_memory: () => void;
}

// Global reference to the loaded WebAssembly module
let wasmModule: QuantizerWasmModule | null = null;

// Initialize WebAssembly module
async function initWasmModule(): Promise<void> {
  if (wasmModule) return;

  try {
    const response = await fetch('/wasm/quantizer.wasm');
    const buffer = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(buffer, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256, maximum: 1024 }),
      },
    });
    wasmModule = module.instance.exports as QuantizerWasmModule;
  } catch (error) {
    console.error('Failed to initialize WebAssembly module:', error);
    throw new Error(`WebAssembly initialization failed: ${error}`);
  }
}

// Rust quantization callback for batch processing
export async function invokeRustQuantizer({
  weights,
  batchSize,
  configs,
}: {
  weights: Float32Array[];
  batchSize: number;
  configs: ExtendedQuantizationConfig[];
}): Promise<Float32Array[]> {
  await initWasmModule();

  if (!wasmModule) {
    throw new Error('WebAssembly module not initialized');
  }

  try {
    // Validate inputs
    if (weights.length !== configs.length) {
      throw new Error('Mismatch between weights and configs length');
    }
    const supportedTechniques = ['gptq', 'qlora', 'awq', 'qat'];
    for (const config of configs) {
      if (!supportedTechniques.includes(config.technique)) {
        throw new Error(`Unsupported quantization technique: ${config.technique}`);
      }
      if (![1, 4, 8].includes(config.bitDepth)) {
        throw new Error(`Unsupported bit depth: ${config.bitDepth}`);
      }
      if (config.technique === 'qlora' || config.technique === 'qat') {
        if (!config.loraRank || config.loraRank <= 0) {
          throw new Error('LoRA rank must be specified and positive');
        }
        if (!config.loraCheckpoint || !config.loraCheckpoint.endsWith('.safetensors')) {
          throw new Error('LoRA checkpoint must be a Safetensors file');
        }
      }
    }

    // Flatten weights for WebAssembly
    const flatWeights = Float32Array.from(weights.flat());
    const results: Float32Array[] = [];

    // Process each batch
    for (let i = 0; i < weights.length; i += batchSize) {
      const batchWeights = flatWeights.slice(
        i * weights[0].length,
        (i + batchSize) * weights[0].length
      );
      const batchConfigs = configs.slice(i, i + batchSize);

      // Use the first config for simplicity
      const config = batchConfigs[0];
      const blockSize = config.blockSize || 128;
      const mode = config.mode || 'symmetric';
      const loraRank = config.loraRank || 8;
      const loraCheckpoint = config.loraCheckpoint || './lora_checkpoints/bart-lora-samsum/adapter_model.safetensors';
      const qatEnabled = config.qatEnabled || false;
      const learningRate = config.learningRate || 0.001;

      const quantizedBatch = wasmModule.quantize_batch(
        batchWeights,
        batchSize,
        config.bitDepth,
        config.technique,
        blockSize,
        mode,
        loraRank,
        loraCheckpoint,
        qatEnabled,
        learningRate
      );

      // Split quantized batch back into individual expert weights
      for (let j = 0; j < batchSize && i + j < weights.length; j++) {
        results.push(
          quantizedBatch.slice(
            j * weights[0].length,
            (j + 1) * weights[0].length
          )
        );
      }
    }

    // Free WebAssembly memory
    wasmModule.free_memory();

    return results;
  } catch (error) {
    console.error('Quantization error:', error);
    throw new Error(`Failed to quantize weights: ${error}`);
  }
}

export class MixedPrecisionQuantizer {
  async quantizeBatch(
    experts: Expert[],
    configs: ExtendedQuantizationConfig[]
  ): Promise<Expert[]> {
    if (experts.length === 0 || configs.length === 0) {
      throw new Error('Empty experts or configs provided');
    }
    if (experts.length !== configs.length) {
      throw new Error('Mismatch between experts and configs length');
    }

    // Extract weights for batch processing
    const weights = experts.map((expert) => {
      if (!expert.weights || expert.weights.length === 0) {
        throw new Error(`Invalid weights for expert ${expert.id}`);
      }
      return expert.weights;
    });

    // Invoke Rust-based quantization
    const quantizedWeights = await invokeRustQuantizer({
      weights,
      batchSize: Math.min(4, experts.length),
      configs,
    });

    // Return updated experts
    return experts.map((expert, i) => ({
      ...expert,
      weights: quantizedWeights[i],
      bitDepth: configs[i].bitDepth,
    }));
  }

  async quantize(expert: Expert, config: QuantizationConfig): Promise<Expert> {
    const extendedConfig: ExtendedQuantizationConfig = {
      ...config,
      mode: 'symmetric',
      loraRank: config.technique === 'qlora' || config.technique === 'qat' ? 8 : undefined,
      loraCheckpoint:
        config.technique === 'qlora' || config.technique === 'qat'
          ? './lora_checkpoints/bart-lora-samsum/adapter_model.safetensors'
          : undefined,
      qatEnabled: config.technique === 'qat',
      learningRate: 0.001,
    };
    const results = await this.quantizeBatch([expert], [extendedConfig]);
    return results[0];
  }
}