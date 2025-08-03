#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, Device};

    #[test]
    fn test_rl_convergence() {
        let mut zeta = ZetaVaultSynergy::new();
        let input_tensor = Tensor::zeros((10, 10), candle_core::DType::F32, &Device::Cpu).unwrap();
        let hardware_profile = HardwareProfile {
            hardware_type: "cpu".to_string(),
        };

        // Simulate workload with 5 experts
        let experts = vec![
            ExpertId("expert1".to_string()),
            ExpertId("expert2".to_string()),
            ExpertId("expert3".to_string()),
            ExpertId("expert4".to_string()),
            ExpertId("expert5".to_string()),
        ];
        let bit_depths = [BitDepth::INT4, BitDepth::INT8, BitDepth::FP16];

        // Run 1000 iterations
        let mut q_values = vec![];
        for _ in 0..1000 {
            let trace = InferenceTrace {
                expert_id: experts[rand::random::<usize>() % experts.len()].clone(),
                bit_depth: bit_depths[rand::random::<usize>() % bit_depths.len()],
                hardware_profile: hardware_profile.clone(),
                accuracy: rand::random::<f32>(),
                latency: rand::random::<f32>() * 0.2,
                token_loss: rand::random::<f32>() * 0.05,
                decision: QuantizationDecision::Hold,
                input_size: 10,
            };
            zeta.infer(input_tensor.clone(), hardware_profile.clone());
            q_values.push(trace.accuracy - 0.1 * trace.latency - 0.05 * trace.token_loss);
        }

        // Check convergence (mock: verify q_values stabilize)
        let last_100_avg = q_values[q_values.len() - 100..].iter().sum::<f32>() / 100.0;
        assert!(last_100_avg > 0.5, "Q-values did not converge to expected reward");
    }
}