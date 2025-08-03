import { RLState, RLAction, RLReward, BitDepth, RLTransition } from './types';

type QTableKey = string;
type QValues = [number, number, number]; // [increase, decrease, maintain]

/**
 * Q-learning implementation for bit-depth optimization
 */
export class QLearning {
  private qTable: Map<QTableKey, QValues> = new Map();
  private lambda: number = 0.1; // Weight for latency in reward calculation
  private learningRate: number = 0.1;
  private explorationRate: number = 0.1;
  private readonly actions: RLAction['type'][] = ['increase', 'decrease', 'maintain'];
  private readonly bitDepths: BitDepth[] = [1, 4, 8];

  constructor() {
    this.selectBitDepth = this.selectBitDepth.bind(this);
    this.updateQTable = this.updateQTable.bind(this);
  }

  /**
   * Selects the optimal bit depth for inference based on the current state
   * @param state Current RL state
   * @returns Promise resolving to the selected bit depth
   */
  public async selectBitDepth(state: RLState): Promise<BitDepth> {
    const stateKey = this.getStateKey(state);
    const qValues = this.getQValues(stateKey);
    
    // Epsilon-greedy action selection
    const action = Math.random() < this.explorationRate
      ? this.getRandomAction()
      : this.getBestAction(qValues);

    return this.applyAction(state.currentBitDepth, action);
  }

  /**
   * Updates the Q-table based on the observed reward
   * @param state The state before taking the action
   * @param action The action taken
   * @param reward The observed reward
   */
  /**
   * Updates the Q-learning model with a new experience
   * @param transition The transition object containing state, action, reward, and nextState
   */
  public async update(transition: RLTransition): Promise<void> {
    const { state, action, reward } = transition;
    // Convert reward number to RLReward interface if needed
    const rlReward: RLReward = typeof reward === 'number' 
      ? {
          accuracy: reward,
          latency: 0, // Default latency
          metadata: {}
        }
      : reward;
    await this.updateQTable(state, action, rlReward);
  }

  /**
   * Updates the Q-table based on the observed reward
   * @param state The state before taking the action
   * @param action The action taken
   * @param reward The observed reward
   * @private
   */
  private async updateQTable(state: RLState, action: RLAction, reward: RLReward): Promise<void> {
    const stateKey = this.getStateKey(state);
    const qValues = this.getQValues(stateKey);
    const actionIdx = this.actions.indexOf(action.type);
    
    if (actionIdx === -1) {
      throw new Error(`Invalid action: ${action.type}`);
    }

    // Calculate reward: accuracy - (latency * lambda) - tokenDrop
    const rewardValue = reward.accuracy - (reward.latency * this.lambda) - (reward.tokenDrop || 0);
    
    // Q-learning update rule: Q(s,a) = Q(s,a) + α * (r - Q(s,a))
    qValues[actionIdx] = qValues[actionIdx] + this.learningRate * (rewardValue - qValues[actionIdx]);
    
    this.qTable.set(stateKey, qValues);
  }

  /**
   * Gets the current Q-values for a state, initializing if not present
   * @param stateKey The state key
   * @returns Array of Q-values for each action
   */
  private getQValues(stateKey: QTableKey): QValues {
    if (!this.qTable.has(stateKey)) {
      // Initialize with small random values to encourage exploration
      const initialValues: QValues = [
        Math.random() * 0.1,
        Math.random() * 0.1,
        Math.random() * 0.1
      ];
      this.qTable.set(stateKey, initialValues);
    }
    return this.qTable.get(stateKey)!;
  }

  /**
   * Converts a state to a string key for the Q-table
   */
  private getStateKey(state: RLState): QTableKey {
    return `${state.expertId}:${state.currentBitDepth}:${state.hardwareClass}`;
  }

  /**
   * Selects a random action (for exploration)
   */
  private getRandomAction(): RLAction['type'] {
    const randomIndex = Math.floor(Math.random() * this.actions.length);
    return this.actions[randomIndex];
  }

  /**
   * Selects the best action based on current Q-values (for exploitation)
   */
  private getBestAction(qValues: QValues): RLAction['type'] {
    const maxQ = Math.max(...qValues);
    const bestActions = this.actions.filter((_, i) => qValues[i] === maxQ);
    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }

  /**
   * Applies an action to the current bit depth
   */
  private applyAction(current: BitDepth, action: RLAction['type']): BitDepth {
    const currentIdx = this.bitDepths.indexOf(current);
    
    if (currentIdx === -1) {
      throw new Error(`Invalid current bit depth: ${current}`);
    }

    switch (action) {
      case 'increase':
        return currentIdx < this.bitDepths.length - 1 
          ? this.bitDepths[currentIdx + 1] 
          : current;
      case 'decrease':
        return currentIdx > 0 
          ? this.bitDepths[currentIdx - 1] 
          : current;
      case 'maintain':
        return current;
      default:
        throw new Error(`Invalid action: ${action}`);
    }
  }
}

// Export a singleton instance
export const qLearning = new QLearning();

// Export the selectBitDepth function for backward compatibility
export const selectBitDepth = qLearning.selectBitDepth.bind(qLearning);