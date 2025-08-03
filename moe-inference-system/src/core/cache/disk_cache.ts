import { compressWithPersistentHomology } from '../compression/persistent_homology';

export class DiskCache {
  private compressionThreshold: number;

  constructor(threshold: number) {
    this.compressionThreshold = threshold;
  }

  async evict(data: Float32Array): Promise<Float32Array> {
    return await compressWithPersistentHomology(data, this.compressionThreshold);
  }
}