use crossbeam_queue::SegQueue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTrace {
    pub expert_id: ExpertId,
    pub bit_depth: BitDepth,
    pub hardware_profile: HardwareProfile,
    pub accuracy: f32,
    pub latency: f32,
    pub token_loss: f32,
    pub decision: QuantizationDecision,
    pub input_size: usize,
}

pub struct InferenceTraceBuffer {
    queue: SegQueue<InferenceTrace>,
}

impl InferenceTraceBuffer {
    pub fn new() -> Self {
        InferenceTraceBuffer {
            queue: SegQueue::new(),
        }
    }

    pub fn append(&self, trace: InferenceTrace) {
        self.queue.push(trace);
    }

    pub fn flush(&self) -> Vec<InferenceTrace> {
        let mut traces = Vec::new();
        while let Some(trace) = self.queue.pop() {
            traces.push(trace);
        }
        traces
    }
}