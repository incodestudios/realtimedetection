import { Keypoint } from '@tensorflow-models/pose-detection';

export function scorePoseSimilarity(
  live: Keypoint[],
  reference: Keypoint[]
): number {
  if (!live || !reference || live.length !== reference.length) return 0;

  let total = 0;
  let count = 0;

  for (let i = 0; i < live.length; i++) {
    const dx = live[i].x - reference[i].x;
    const dy = live[i].y - reference[i].y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    total += dist;
    count++;
  }

  const normalized = 100 - Math.min((total / count) / 10, 100);
  return normalized;
}