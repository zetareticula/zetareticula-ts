import { Expert } from '../moe/types';

/**
 * Mock experts for development and testing
 */

export const expert1: Expert = {
  id: 'expert-1',
  bitDepth: 8,
  weights: new Float32Array([0.2, 0.3, 0.1, 0.4]),
  metadata: {
    name: 'Vision Transformer',
    description: 'Specialized in image classification tasks',
    inputSize: 224,
    numParameters: 86_000_000,
    defaultPrecision: 'mixed'
  }
};

export const expert2: Expert = {
  id: 'expert-2',
  bitDepth: 4,
  weights: new Float32Array([0.1, 0.4, 0.2, 0.3]),
  metadata: {
    name: 'BERT Base',
    description: 'Specialized in natural language understanding',
    maxSequenceLength: 512,
    numParameters: 110_000_000,
    defaultPrecision: 'int8'
  }
};

// Additional mock experts for testing
export const expert3: Expert = {
  id: 'expert-3',
  bitDepth: 1,
  weights: new Float32Array([0.3, 0.2, 0.4, 0.1]),
  metadata: {
    name: 'EfficientNet',
    description: 'Lightweight image classification model',
    inputSize: 300,
    numParameters: 5_300_000,
    defaultPrecision: 'binary'
  }
};

// Export all experts as an array
export const allExperts: Expert[] = [expert1, expert2, expert3];
