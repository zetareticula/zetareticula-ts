import { MoE } from '../core/moe/expert';
import { Expert } from '../core/moe/types';

describe('MoE System', () => {
  const mockExperts: Expert[] = [
    {
      id: 'expert1',
      bitDepth: 8,
      weights: new Float32Array([0.1, 0.2, 0.3])
    },
    {
      id: 'expert2',
      bitDepth: 4,
      weights: new Float32Array([0.4, 0.5, 0.6])
    }
  ];

  let moe: MoE;

  beforeEach(() => {
    moe = new MoE(mockExperts);
  });

  it('should initialize with experts', () => {
    expect(moe).toBeInstanceOf(MoE);
  });

  it('should route input to the most suitable expert', async () => {
    const input = {
      features: new Float32Array([0.1, 0.2, 0.3]),
      hardwareClass: 'cpu' as const
    };

    const result = await moe.routeInput(input);
    expect(result).toBeDefined();
    expect(mockExperts.some(e => e.id === result.id)).toBe(true);
  });

  it('should handle empty experts array', () => {
    expect(() => new MoE([])).toThrow('At least one expert is required');
  });
});
