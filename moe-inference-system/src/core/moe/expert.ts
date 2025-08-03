import { Expert, MoEInput } from './types';
import { softmaxGating } from './gating';
import { QLearning } from '../rl/q_learning';
import { createLogger } from '../logger';
import { RLState, BitDepth } from '../rl/types';

export class MoE {
  private experts: Expert[];
  private logger = createLogger('MoE');
  private qLearning: QLearning;

  // Track inference metrics for RL
  private inferenceMetrics: Map<string, {
    startTime: number;
    inputLength: number;
    expertId: string;
    bitDepth: BitDepth;
  }> = new Map();

  constructor(experts: Expert[]) {
    if (!experts || experts.length === 0) {
      throw new Error('At least one expert is required');
    }
    this.experts = experts;
    this.qLearning = new QLearning();
    this.logger.info(`Initialized MoE with ${experts.length} experts`);
  }

  async routeInput(input: MoEInput, requestId?: string): Promise<Expert> {
    const inferenceId = requestId || `inf_${Date.now()}`;
    const startTime = Date.now();

    try {
      if (!input.features || input.features.length === 0) {
        throw new Error('Input features cannot be empty');
      }

      this.logger.debug('Routing input', { 
        inferenceId,
        inputLength: input.features.length,
        hardwareClass: input.hardwareClass 
      });

      // Get expert selection
      const gating = softmaxGating(input, this.experts);
      const topExpert = gating.reduce((prev, curr) =>
        curr.probability > prev.probability ? curr : prev
      );

      this.logger.debug('Selected expert', { 
        inferenceId,
        expertId: topExpert.expertId,
        probability: topExpert.probability.toFixed(4)
      });

      // Find the expert
      const expert = this.experts.find(e => e.id === topExpert.expertId);
      if (!expert) {
        throw new Error(`Expert ${topExpert.expertId} not found`);
      }

          // Store inference start metrics for RL
      this.inferenceMetrics.set(inferenceId, {
        startTime,
        inputLength: input.features.length,
        expertId: expert.id,
        bitDepth: expert.bitDepth as BitDepth
      });

      // Get bit depth from RL
      const state: RLState = {
        expertId: expert.id,
        currentBitDepth: expert.bitDepth,
        hardwareClass: input.hardwareClass
      };

      const bitDepth = await this.qLearning.selectBitDepth(state);

      this.logger.info('Inference started', { 
        inferenceId,
        expertId: expert.id,
        bitDepth,
        hardwareClass: input.hardwareClass 
      });

      return { ...expert, bitDepth };
    } catch (error) {
      this.logger.error('Error in routeInput:', { 
        inferenceId, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      throw error;
    }
  }

  /**
   * Report inference results for reinforcement learning
   * @param inferenceId The ID of the inference request
   * @param accuracy The accuracy of the inference (0-1)
   * @param tokenDrop Number of tokens dropped (if any)
   */
  async reportInferenceResult(
    inferenceId: string, 
    accuracy: number, 
    tokenDrop: number = 0
  ): Promise<void> {
    const metrics = this.inferenceMetrics.get(inferenceId);
    if (!metrics) {
      this.logger.warn('No metrics found for inference', { inferenceId });
      return;
    }

    const latency = Date.now() - metrics.startTime;
    
    const state: RLState = {
      expertId: metrics.expertId,
      currentBitDepth: metrics.bitDepth,
      hardwareClass: 'cpu' // Simplified for this example
    };

    // In a real implementation, we would determine the action based on the bit depth change
    const action = { type: 'maintain' as const };
    
    const reward = {
      accuracy,
      latency,
      tokenDrop
    };

    try {
      await this.qLearning.updateQTable(state, action, reward);
      this.logger.debug('Updated Q-table', { inferenceId, state, action, reward });
    } catch (error) {
      this.logger.error('Failed to update Q-table', { 
        inferenceId, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    } finally {
      this.inferenceMetrics.delete(inferenceId);
    }
  }
}