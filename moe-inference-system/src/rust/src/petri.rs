use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PetriPlace {
    ExpertDispatch,
    QuantizeKV,
    Infer,
    CompressResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PetriTransition {
    pub from: PetriPlace,
    pub to: PetriPlace,
    pub weight: u32,
}

pub struct PetriNetMonoid {
    transitions: Vec<PetriTransition>,
}

impl PetriNetMonoid {
    pub fn new() -> Self {
        PetriNetMonoid {
            transitions: Vec::new(),
        }
    }

    pub fn log_transition(&mut self, from: PetriPlace, to: PetriPlace, bit_depth: BitDepth) {
        let weight = match bit_depth {
            BitDepth::INT4 => 1,
            BitDepth::INT8 => 2,
            BitDepth::FP16 => 3,
        };
        self.transitions.push(PetriTransition { from, to, weight });
    }

    pub fn get_transitions(&self) -> &[PetriTransition] {
        &self.transitions
    }
}