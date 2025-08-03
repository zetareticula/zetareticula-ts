import { RLState, RLAction, RLReward } from './types';

export class QLearning {
  private qTable: Map<string, number[]> = new Map();
  private lambda: number = 0.1;

  async selectBitDepth(state: RLState): Promise<BitDepth> {
    const stateKey = `${state.expertId}:${state.currentBitDepth}:${state.hardwareClass}`;
    const actions = ['increase', 'decrease', 'maintain'];
    const qValues = this.qTable.get(stateKey) || actions.map(() => 0);

    // Epsilon-greedy action selection
    const epsilon = 0.1;
    const action = Math.random() < epsilon
      ? actions[Math.floor(Math.random() * actions.length)]
      : actions[qValues.indexOf(Math.max(...qValues))];

    return this.applyAction(state.currentBitDepth, action);
  }

  async updateQTable(state: RLState, action: RLAction, reward: RLReward): Promise<void> {
    const stateKey = `${state.expertId}:${state.currentBitDepth}:${state.hardwareClass}`;
    const qValues = this.qTable.get(stateKey) || [0, 0, 0];
    const actionIdx = ['increase', 'decrease', 'maintain'].indexOf(action.type);
    const newQ = reward.accuracy - (reward.latency * this.lambda) - reward.tokenDrop;
    qValues[actionIdx] = qValues[actionIdx] + 0.1 * (newQ - qValues[actionIdx]);
    this.qTable.set(stateKey, qValues);
  }

  private applyAction(current: BitDepth, action: string): BitDepth {
    const depths: BitDepth[] = [1, 4, 8];
    const idx = depths.indexOf(current);
    if (action === 'increase' && idx < depths.length - 1) return depths[idx + 1];
    if (action === 'decrease' && idx > 0) return depths[idx - 1];
    return current;
  }
}