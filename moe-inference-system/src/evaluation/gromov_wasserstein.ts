/**
 * Gromov–Wasserstein alignment stub
 * ----------------------------------
 * Provides a typed placeholder implementation for computing the
 * Gromov–Wasserstein distance between two sets of embeddings.
 *
 * This is a heavy optimal-transport problem; in production you may:
 *   • Offload to a Python micro-service (POT, geomloss, etc.)
 *   • Use WebAssembly-compiled OT solvers
 *   • Approximate via Sinkhorn iterations in tfjs
 *
 * For now we export a synchronous stub that returns 0 and logs a warning.
 */

export interface EmbeddingMatrix {
  /** Shape: (n_samples, dim) */
  data: Float32Array | number[][];
  modality: string;
}

export interface GWOptions {
  /**
   * Regularisation strength for Sinkhorn iterations. 0 → exact OT (slow).
   */
  epsilon?: number;
  /** Maximum Sinkhorn iterations. */
  maxIter?: number;
  /** Convergence tolerance. */
  tol?: number;
}

/**
 * Compute (placeholder) Gromov–Wasserstein distance between X and Y.
 * @param X Embeddings from modality A.
 * @param Y Embeddings from modality B.
 * @returns Distance (0 → identical). Always 0 in stub.
 */
export function gromovWassersteinDistance(
  X: EmbeddingMatrix,
  Y: EmbeddingMatrix,
  options: GWOptions = {}
): number {
  /* eslint-disable no-console */
  try {
    const { spawnSync } = require('node:child_process');
    const script = process.env.GW_PY_SCRIPT || __dirname + '/../../scripts/gw_distance.py';
    const payload = JSON.stringify({ X: embedToArray(X), Y: embedToArray(Y), ...options });
    const res = spawnSync('python3', [script], { input: payload, encoding: 'utf-8' });
    if (res.error) throw res.error;
    if (res.status !== 0) throw new Error(res.stderr);
    const dist = parseFloat(res.stdout.trim());
    return Number.isFinite(dist) ? dist : Number.POSITIVE_INFINITY;
  } catch (err) {
    /* eslint-disable no-console */
    console.warn('[GW] Python solver unavailable, falling back to 0. Install POT and set GW_PY_SCRIPT.');
    console.debug(err);
    return 0;
  }
}

function embedToArray(mat: EmbeddingMatrix): number[][] {
  if (Array.isArray(mat.data)) return mat.data as number[][];
  // If flat Float32Array, assume caller passes rows externally; fallback to simple chunk by dim guess = 0
  return [];
}

function getSampleCount(mat: EmbeddingMatrix): number {
  if (Array.isArray(mat.data)) return mat.data.length;
  return mat.data.length ? mat.data.length / inferDim(mat) : 0;
}

function inferDim(mat: EmbeddingMatrix): number {
  if (Array.isArray(mat.data)) return mat.data[0]?.length ?? 0;
  // Assuming row-major contiguous flat Float32Array
  return 0; // cannot infer without n_samples
}
