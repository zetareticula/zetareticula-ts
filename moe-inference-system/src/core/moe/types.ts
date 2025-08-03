export interface ExpertMetadata {
  name: string;
  description: string;
  [key: string]: any; // Allow for additional metadata fields
}

export interface Expert {
  /** Unique identifier for the expert */
  id: string;
  
  /** Bit depth for model weights (1, 4, or 8 bits) */
  bitDepth: 1 | 4 | 8;
  
  /** Model weights as a Float32Array */
  weights: Float32Array;
  
  /** Additional metadata about the expert */
  metadata?: ExpertMetadata;
}
  
  export interface GatingOutput {
    expertId: string;
    probability: number;
  }
  
  export interface MoEInput {
    features: Float32Array;
    hardwareClass: 'cpu' | 'gpu' | 'tpu';
  }