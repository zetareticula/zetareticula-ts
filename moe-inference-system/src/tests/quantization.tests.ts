import { invokeRustQuantizer, MixedPrecisionQuantizer } from '../core/quantization/mixed_precision';
import { Expert } from '../core/moe/types';

describe('MixedPrecisionQuantizer', () => {
  it('quantizes with QLoRA and Safetensors checkpoint', async () => {
    const experts: Expert[] = [
      { id: 'expert1', bitDepth: 4, weights: new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) },
      { id: 'expert2', bitDepth: 8, weights: new Float32Array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]) },
    ];
    const configs = [
      {
        technique: 'qlora' as const,
        bitDepth: 4 as const,
        mode: 'symmetric' as const,
        blockSize: 4,
        loraRank: 8,
        loraCheckpoint: './lora_checkpoints/bart-lora-samsum/adapter_model.safetensors',
        qatEnabled: false,
      },
      {
        technique: 'qat' as const,
        bitDepth: 8 as const,
        mode: 'asymmetric' as const,
        blockSize: 4,
        loraRank: 8,
        loraCheckpoint: './lora_checkpoints/bart-lora-samsum/adapter_model.safetensors',
        qatEnabled: true,
        learningRate: 0.001,
      },
    ];

    const quantizer = new MixedPrecisionQuantizer();
    const quantizedExperts = await quantizer.quantizeBatch(experts, configs);

    expect(quantizedExperts).toHaveLength(2);
    expect(quantizedExperts[0].weights).toBeInstanceOf(Float32Array);
    expect(quantizedExperts[0].bitDepth).toBe(4);
    expect(quantizedExperts[1].bitDepth).toBe(8);
    expect(quantizedExperts[0].weights.every((v: number) => Math.abs(v) <= 1)).toBe(true);
    expect(quantizedExperts[1].weights.every((v: number) => Math.abs(v) <= 2)).toBe(true);
  });

  it('throws error for missing Safetensors checkpoint', async () => {
    await expect(
      invokeRustQuantizer({
        weights: [new Float32Array([1.0, 2.0, 3.0, 4.0])],
        batchSize: 1,
        configs: [
          {
            technique: 'qlora',
            bitDepth: 4,
            mode: 'symmetric',
            loraRank: 8,
            loraCheckpoint: './nonexistent.safetensors',
          },
        ],
      })
    ).rejects.toThrow('LoRA checkpoint must be a Safetensors file');
  });

  it('throws error for invalid LoRA rank', async () => {
    await expect(
      invokeRustQuantizer({
        weights: [new Float32Array([1.0, 2.0, 3.0, 4.0])],
        batchSize: 1,
        configs: [
          {
            technique: 'qlora',
            bitDepth: 4,
            mode: 'symmetric',
            loraRank: 0,
            loraCheckpoint: './lora_checkpoints/bart-lora-samsum/adapter_model.safetensors',
          },
        ],
      })
    ).rejects.toThrow('LoRA rank must be specified and positive');
  });
});