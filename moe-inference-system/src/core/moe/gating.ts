import { Expert, GatingOutput, MoEInput } from './types';

// Softmax gating function for expert routing
export function softmaxGating(input: MoEInput, experts: Expert[]): GatingOutput[] {
  const scores = experts.map(expert => {
    // Simplified scoring (Rust handles heavy computation)
    const score = input.features.reduce((sum, val, i) => sum + val * expert.weights[i], 0);
    return { expertId: expert.id, score };
  });

  const expScores = scores.map(s => Math.exp(s.score));
  const sumExpScores = expScores.reduce((sum, val) => sum + val, 0);

  return scores.map((s, i) => ({
    expertId: s.expertId,
    probability: expScores[i] / sumExpScores,
  }));
}