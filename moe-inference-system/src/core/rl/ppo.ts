import { RLState, RLAction } from './types';
import { invokeRustPPOPolicy } from '../../rust';

export class PPO {
  async selectBitDepth(state: RLState): Promise<BitDepth> {
    const policyOutput = await invokeRustPPOPolicy(state);
    return policyOutput.bitDepth;
  }
}