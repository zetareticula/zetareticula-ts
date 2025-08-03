import { expectEmbeddingsToAlign } from '../../src/evaluation/test_utils';
import { EmbeddingMatrix } from '../../src/evaluation/gromov_wasserstein';

import * as math from 'mathjs';

function randomEmbeddings(n: number, dim: number): number[][] {
  return Array.from({ length: n }, () => (
    Array.from({ length: dim }, () => Math.random())
  ));
}

describe('Gromov–Wasserstein alignment', () => {
  it('returns small distance for identical embeddings', () => {
    const X: EmbeddingMatrix = { data: randomEmbeddings(50, 32), modality: 'text' };
    // Create Y by applying a random orthonormal rotation and Gaussian noise
const Xmat = math.matrix(X.data as number[][]);
const R = math.qr(math.random([dim, dim], -1, 1)).Q as math.Matrix; // orthonormal
const Ymat = math.add(math.multiply(Xmat, R), math.random([50, dim], -0.01, 0.01)) as math.Matrix;
const Y: EmbeddingMatrix = { data: Ymat.toArray() as number[][], modality: 'image' };
    expectEmbeddingsToAlign(X, Y, 0.2);
  });
});
