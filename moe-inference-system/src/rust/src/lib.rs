mod quantization;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init() {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
}