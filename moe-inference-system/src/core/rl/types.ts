/**
 * Supported bit depths for model quantization
 */
export type BitDepth = 1 | 4 | 8;

/**
 * Represents the state in the RL environment
 */
export interface RLState {
  /** Unique identifier for the expert model */
  expertId: string;
  /** Current bit depth of the model */
  currentBitDepth: BitDepth;
  /** Hardware class the model is running on */
  hardwareClass: 'cpu' | 'gpu' | 'tpu';
  /** Optional: Additional context that might affect the decision */
  context?: Record<string, any>;
}

/**
 * Possible actions the RL agent can take
 */
export interface RLAction {
  type: 'increase' | 'decrease' | 'maintain';
  /** Optional: Additional action parameters */
  params?: Record<string, any>;
}

/**
 * Reward signal for the RL agent
 */
export interface RLReward {
  /** Model accuracy (0-1) */
  accuracy: number;
  /** Inference latency in milliseconds */
  latency: number;
  /** Number of tokens dropped (if any) */
  tokenDrop?: number;
  /** Optional: Additional reward components */
  metadata?: {
    /** Peak memory usage in MB */
    memoryUsage?: number;
    /** Energy consumption in joules */
    energyUsage?: number;
    /** Any other relevant metrics */
    [key: string]: any;
  };
}

/**
 * Represents a state transition in the RL environment
 */
export interface RLTransition {
  /** The state before taking the action */
  state: RLState;
  /** The action taken */
  action: RLAction;
  /** The reward received */
  reward: RLReward | number;
  /** The next state after taking the action */
  nextState: RLState;
  /** Optional metadata about the transition */
  metadata?: Record<string, any>;
}

/**
 * Configuration for the Q-learning algorithm
 */
export interface QLearningConfig {
  /** Learning rate (alpha) */
  learningRate?: number;
  /** Discount factor (gamma) */
  discountFactor?: number;
  /** Exploration rate (epsilon) */
  explorationRate?: number;
  /** Weight for latency in reward calculation */
  lambda?: number;
  /** Decay rate for exploration */
  explorationDecay?: number;
  /** Minimum exploration rate */
  minExplorationRate?: number;
}