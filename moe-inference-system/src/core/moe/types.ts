export interface Expert {
    id: string;
    bitDepth: 1 | 4 | 8;
    weights: Float32Array;
  }
  
  export interface GatingOutput {
    expertId: string;
    probability: number;
  }
  
  export interface MoEInput {
    features: Float32Array;
    hardwareClass: 'cpu' | 'gpu' | 'tpu';
  }