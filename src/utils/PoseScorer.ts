import { Keypoint } from '@tensorflow-models/pose-detection';

function getAngle(a: Keypoint, b: Keypoint, c: Keypoint): number {
  const ab = [a.x - b.x, a.y - b.y];
  const cb = [c.x - b.x, c.y - b.y];
  const dot = ab[0] * cb[0] + ab[1] * cb[1];
  const magAB = Math.hypot(...ab);
  const magCB = Math.hypot(...cb);
  const cosine = dot / (magAB * magCB);
  return Math.acos(Math.min(Math.max(cosine, -1), 1)) * (180 / Math.PI);
}

export function scorePoseByAngles(
  live: (Keypoint | undefined)[],
  ref: (Keypoint | undefined)[]
): number {
  const joints: [number, number, number][] = [
    [5, 7, 9],   // left shoulder → elbow → wrist
    [6, 8, 10],  // right shoulder → elbow → wrist
    [11, 13, 15], // left hip → knee → ankle
    [12, 14, 16]  // right hip → knee → ankle
  ];

  let totalDiff = 0;
  let valid = 0;

  for (const [a, b, c] of joints) {
    const la = live[a];
    const lb = live[b];
    const lc = live[c];
    const ra = ref[a];
    const rb = ref[b];
    const rc = ref[c];

    const allLive = la && lb && lc;
    const allRef = ra && rb && rc;

    if (
      allLive &&
      allRef &&
      la.score !== undefined && lb.score !== undefined && lc.score !== undefined &&
      ra.score !== undefined && rb.score !== undefined && rc.score !== undefined &&
      la.score > 0.4 && lb.score > 0.4 && lc.score > 0.4 &&
      ra.score > 0.4 && rb.score > 0.4 && rc.score > 0.4
    ) {
      const angleLive = getAngle(la, lb, lc);
      const angleRef = getAngle(ra, rb, rc);
      totalDiff += Math.abs(angleLive - angleRef);
      valid++;
    }
  }

  if (valid === 0) return 0;
  const avgDiff = totalDiff / valid;
  return 100 - Math.min(avgDiff, 100); // Scale to 0–100
}