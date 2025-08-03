import { Expert, MoEInput } from './types';
import { softmaxGating } from './gating';
import { selectBitDepth } from '../rl/q_learning';

export class MoE {
  private experts: Expert[];

  constructor(experts: Expert[]) {
    this.experts = experts;
  }

  async routeInput(input: MoEInput): Promise<Expert> {
    const gating = softmaxGating(input, this.experts);
    const topExpert = gating.reduce((prev, curr) =>
      curr.probability > prev.probability ? curr : prev
    );

    // Adjust bit depth using RL
    const bitDepth = await selectBitDepth({
      expertId: topExpert.expertId,
      currentBitDepth: this.experts.find(e => e.id === topExpert.expertId)!.bitDepth,
      hardwareClass: input.hardwareClass,
    });

    return { ...this.experts.find(e => e.id === topExpert.expertId)!, bitDepth };
  }
}