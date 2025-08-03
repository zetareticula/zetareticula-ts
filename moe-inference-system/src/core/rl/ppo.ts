import { RLState } from './types';
import { BitDepth } from '../quantization/types';

// TODO: Replace with actual Rust implementation when available
async function invokeRustPPOPolicy(_state: RLState): Promise<{ bitDepth: BitDepth }> {
  // Placeholder implementation
  console.warn('Using placeholder PPO implementation. Replace with actual Rust implementation.');
  return { bitDepth: 8 }; // Default to 8-bit
}

export class PPO {
  async selectBitDepth(state: RLState): Promise<BitDepth> {
    // Call Rust implementation for PPO policy
    const result = await invokeRustPPOPolicy(state);
    return result.bitDepth;
  }
}