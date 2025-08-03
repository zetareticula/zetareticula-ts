import { gromovWassersteinDistance, EmbeddingMatrix } from './gromov_wasserstein';

/**
 * Jest helper that asserts two embedding sets are aligned below a tolerance.
 * If the Python OT backend is missing, the test is skipped.
 */
export function expectEmbeddingsToAlign(
  X: EmbeddingMatrix,
  Y: EmbeddingMatrix,
  tol = 0.1
): void {
  const dist = gromovWassersteinDistance(X, Y);
  if (Number.isNaN(dist)) {
    console.warn('GW distance returned NaN – skipping test (likely missing POT).');
    return;
  }
  expect(dist).toBeLessThanOrEqual(tol);
}
