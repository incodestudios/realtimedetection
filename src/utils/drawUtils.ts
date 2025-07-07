import * as poseDetection from '@tensorflow-models/pose-detection';

type Keypoint = poseDetection.Keypoint;

const CONNECTED_PAIRS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // right arm
  [5, 6], [6, 7], [7, 8],               // left arm
  [5, 11], [6, 12], [11, 12],           // shoulders
  [11, 13], [13, 15],                   // right leg
  [12, 14], [14, 16],                   // left leg
];

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[]
): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = 'lime';
  ctx.lineWidth = 2;

  for (const [i, j] of CONNECTED_PAIRS) {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    if ((kp1?.score ?? 0) > 0.4 && (kp2?.score ?? 0) > 0.4) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  }
}
