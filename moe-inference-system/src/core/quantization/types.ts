export type BitDepth = 1 | 4 | 8;
export type QuantizationMode = 'symmetric' | 'asymmetric';
export type QuantizationTechnique = 'gptq' | 'qlora' | 'awq' | 'qat';

export interface Quantrect: QuantizationConfig {
  technique: QuantizationTechnique;
  bitDepth: BitDepth;
  blockSize?: number;
}