use crate::policy_engine::{BitPrecisionPolicy, QLearningPolicy, PPOPolicy, QuantizationDecision, ExpertId, BitDepth, HardwareProfile};
use crate::trace_buffer::InferenceTrace;
use candle_core::Tensor;

pub struct RLOptimizer {
    q_learning: QLearningPolicy,
    ppo: PPOPolicy,
}

impl RLOptimizer {
    pub fn new(lambda1: f32, lambda2: f32) -> Self {
        RLOptimizer {
            q_learning: QLearningPolicy::new(lambda1, lambda2, 0.1),
            ppo: PPOPolicy::new(lambda1, lambda2),
        }
    }

    pub fn optimize_bit_depth(
        &mut self,
        trace: InferenceTrace,
        input_tensor: &Tensor,
        hardware_profile: &HardwareProfile,
    ) -> QuantizationDecision {
        // Update both policies
        self.q_learning.update_policy(trace.clone());
        self.ppo.update_policy(trace.clone());

        // Select action using Q-learning (or switch to PPO based on config)
        let experts = self.q_learning.select_experts(input_tensor, hardware_profile);
        experts
            .into_iter()
            .find(|(id, _)| id.0 == trace.expert_id.0)
            .map(|(_, bit_depth)| match (trace.bit_depth, bit_depth) {
                (BitDepth::INT4, BitDepth::INT8) | (BitDepth::INT4, BitDepth::FP16) => QuantizationDecision::Up,
                (BitDepth::INT8, BitDepth::INT4) | (BitDepth::FP16, BitDepth::INT4) => QuantizationDecision::Down,
                _ => QuantizationDecision::Hold,
            })
            .unwrap_or(QuantizationDecision::Hold)
    }
}