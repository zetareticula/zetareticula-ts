use ripser::{ripser, PersistenceDiagram};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub threshold: f32,
}

pub fn compress_with_persistent_homology(data: Vec<f32>, config: CompressionConfig) -> Vec<f32> {
    let diagram = ripser(&data).unwrap_or(PersistenceDiagram::default());
    diagram
        .into_iter()
        .filter(|p| p.persistence() > config.threshold)
        .flat_map(|p| vec![p.birth, p.death])
        .collect()
}