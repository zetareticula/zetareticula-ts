use wasm_bindgen::prelude::*;
use ripser::{ripser, PersistenceDiagram};

#[wasm_bindgen]
pub fn compress_ph(data: Vec<f32>, threshold: f32) -> Result<Vec<f32>, JsValue> {
    let diagram = ripser(&data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let filtered = diagram
        .into_iter()
        .filter(|p| p.persistence() > threshold)
        .flat_map(|p| vec![p.birth, p.death])
        .collect();
    Ok(filtered)
}