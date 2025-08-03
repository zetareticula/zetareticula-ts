import { invokeRustPHCompression } from '../../rust';

export async function compressWithPersistentHomology(
  data: Float32Array,
  threshold: number
): Promise<Float32Array> {
  return await invokeRustPHCompression(data, threshold);
}