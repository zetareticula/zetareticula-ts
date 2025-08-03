export interface RLState {
    expertId: string;
    currentBitDepth: BitDepth;
    hardwareClass: string;
  }
  
  export interface RLAction {
    type: 'increase' | 'decrease' | 'maintain';
  }
  
  export interface RLReward {
    accuracy: number;
    latency: number;
    tokenDrop: number;
  }