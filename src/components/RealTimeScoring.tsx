'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
import referenceKeypoints from '../data/squat_demo_keypoints.json';
import { drawSkeleton } from '../utils/drawUtils';
import { scorePoseByAngles } from '../utils/PoseScorer';
import ScoreChart from './ScoreChart';

const MODEL = poseDetection.SupportedModels.MoveNet;

// ✅ Helper to draw a bounding box
function drawFixedBoundingBox(
  ctx: CanvasRenderingContext2D,
  box: { x: number; y: number; width: number; height: number }
) {
  ctx.save();
  ctx.strokeStyle = 'lime';
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.restore();
}

const RealTimeScoring: React.FC = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [score, setScore] = useState<number | null>(null);
  const [scoreHistory, setScoreHistory] = useState<{ time: number; score: number }[]>([]);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [status, setStatus] = useState<'Detecting' | 'Ready'>('Detecting');
  const [showPlusTwo, setShowPlusTwo] = useState(false);
  const boundingBoxRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);

  useEffect(() => {
    const init = async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      const detector = await poseDetection.createDetector(MODEL, {
        modelType: 'SinglePose.Lightning',
      });
      setDetector(detector);
    };
    init();
  }, []);

  useEffect(() => {
    if (!detector) return;

    const waitForRefs = setInterval(() => {
      const webcamVideo = webcamRef.current?.video;
      const referenceVideo = videoRef.current;
      const canvas = webcamCanvasRef.current;

      if (webcamVideo?.readyState === 4 && referenceVideo?.readyState === 4 && canvas) {
        const width = canvas.width;
        const height = canvas.height;

        // ✅ Dynamically center the bounding box (50% center)
        const boxWidth = 320;
        const boxHeight = 360;
        boundingBoxRef.current = {
          x: (width - boxWidth) / 2,
          y: (height - boxHeight) / 2,
          width: boxWidth,
          height: boxHeight,
        };

        clearInterval(waitForRefs);
        startScoringLoop();
      }
    }, 100);

    const startScoringLoop = () => {
      const interval = setInterval(async () => {
        const webcamVideo = webcamRef.current?.video as HTMLVideoElement;
        const referenceVideo = videoRef.current;

        if (!webcamVideo || !referenceVideo || !boundingBoxRef.current) return;

        const [livePose] = await detector.estimatePoses(webcamVideo);
        const frameIdx = Math.floor(referenceVideo.currentTime * 5);
        const refKeypoints = referenceKeypoints[frameIdx];
        const boundingBox = boundingBoxRef.current;

        if (livePose?.keypoints) {
          const validKeypoints = livePose.keypoints.filter(
            (kp) => kp.score !== undefined && kp.score > 0.4
          );

          const insideBox = validKeypoints.every(
            (kp) =>
              kp.x >= boundingBox.x &&
              kp.x <= boundingBox.x + boundingBox.width &&
              kp.y >= boundingBox.y &&
              kp.y <= boundingBox.y + boundingBox.height
          );

          setStatus(insideBox ? 'Ready' : 'Detecting');

          if (refKeypoints) {
            const liveScore = scorePoseByAngles(livePose.keypoints, refKeypoints);
            setScore(liveScore);
            setScoreHistory((prev) => [
              ...prev.slice(-50),
              { time: referenceVideo.currentTime, score: liveScore },
            ]);

            if (liveScore >= 80) {
              setShowPlusTwo(true);
              setTimeout(() => setShowPlusTwo(false), 1000);
            }

            const webcamCtx = webcamCanvasRef.current?.getContext('2d');
            if (webcamCtx) {
              webcamCtx.clearRect(0, 0, webcamCtx.canvas.width, webcamCtx.canvas.height);
              drawSkeleton(webcamCtx, livePose.keypoints);
              drawFixedBoundingBox(webcamCtx, boundingBox);
            }

            const videoCtx = videoCanvasRef.current?.getContext('2d');
            if (videoCtx) {
              videoCtx.clearRect(0, 0, videoCtx.canvas.width, videoCtx.canvas.height);
              drawSkeleton(videoCtx, refKeypoints);
            }
          }
        } else {
          setStatus('Detecting');
        }
      }, 200);

      return () => clearInterval(interval);
    };

    return () => clearInterval(waitForRefs);
  }, [detector]);

  return (
    <div className="flex w-screen h-screen overflow-hidden relative font-sans">
      {/* Left: Video */}
      <div className="relative w-1/2 h-full bg-black">
        <video
          ref={videoRef}
          src="/Video_Request_Single_Leg_Stand.mp4"
          autoPlay
          muted
          loop
          className="w-full h-full object-cover"
        />
        <canvas
          ref={videoCanvasRef}
          width={640}
          height={480}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Right: Webcam */}
      <div className="relative w-1/2 h-full bg-black">
        <Webcam
          ref={webcamRef}
          mirrored
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 640, height: 480 }}
          className="w-full h-full object-cover"
        />
        <canvas
          ref={webcamCanvasRef}
          width={640}
          height={480}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />

        {/* +2 */}
        {showPlusTwo && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-[64px] font-extrabold text-pink-500 animate-bounce z-50">
            +2
          </div>
        )}

        {/* Status */}
        <div className="absolute top-2 left-2 bg-white bg-opacity-80 px-3 py-1 rounded text-black font-semibold text-sm z-50">
          {status === 'Ready' ? '✅ Ready' : '🔎 Detecting...'}
        </div>

        {/* Score */}
        <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white px-3 py-1 rounded text-sm font-semibold z-50">
          Score: {score !== null ? score.toFixed(2) : '...'}
        </div>
      </div>

      {/* Chart */}
      <div className="absolute bottom-4 left-4 w-[300px] h-[150px] bg-white bg-opacity-80 rounded p-2 z-50">
        <ScoreChart data={scoreHistory} />
      </div>
    </div>
  );
};

export default RealTimeScoring;