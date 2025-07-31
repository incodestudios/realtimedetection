import React, { useEffect, useRef, useState } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { scorePoseByAngles } from '../utils/PoseScorer';
import { drawSkeleton } from '../utils/drawUtils';
import referenceKeypoints from '../data/squat_demo_keypoints.json';
import Webcam from 'react-webcam';
import ScoreChart from './ScoreChart';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';

const MODEL = poseDetection.SupportedModels.MoveNet;

const RealTimeScoring: React.FC = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [score, setScore] = useState<number | null>(null);
  const [scoreHistory, setScoreHistory] = useState<{ time: number; score: number }[]>([]);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [status, setStatus] = useState<'Detecting' | 'Ready'>('Detecting'); // 👈 added status

  // Load pose detection model once
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

  // Start scoring loop after detector is ready
  useEffect(() => {
    if (!detector) return;

    const waitForRefs = setInterval(() => {
      const webcamVideo = webcamRef.current?.video;
      const referenceVideo = videoRef.current;

      if (
        webcamVideo?.readyState === 4 &&
        referenceVideo?.readyState === 4
      ) {
        clearInterval(waitForRefs);
        startScoringLoop();
      }
    }, 100);

    const startScoringLoop = () => {
      const interval = setInterval(async () => {
        const webcamVideo = webcamRef.current?.video as HTMLVideoElement;
        const referenceVideo = videoRef.current;

        if (!webcamVideo || !referenceVideo) return;

        const [livePose] = await detector.estimatePoses(webcamVideo);
        const frameIdx = Math.floor(referenceVideo.currentTime * 5); // assuming 5 FPS
        const refKeypoints = referenceKeypoints[frameIdx];

        if (livePose?.keypoints) {
          const validKeypoints = livePose.keypoints.filter(
            kp => kp.score !== undefined && kp.score > 0.4
          );

          setStatus(validKeypoints.length > 5 ? 'Ready' : 'Detecting'); // 👈 update status

          if (refKeypoints) {
            const liveScore = scorePoseByAngles(livePose.keypoints, refKeypoints);
            setScore(liveScore);
            setScoreHistory(prev => [
              ...prev.slice(-50),
              { time: referenceVideo.currentTime, score: liveScore }
            ]);

            const webcamCtx = webcamCanvasRef.current?.getContext('2d');
            if (webcamCtx) drawSkeleton(webcamCtx, livePose.keypoints);

            const videoCtx = videoCanvasRef.current?.getContext('2d');
            if (videoCtx) drawSkeleton(videoCtx, refKeypoints);
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
<div className="flex flex-col items-center gap-6">
  <h2 className="text-xl font-semibold">Real-Time Movement Scoring</h2>

  {/* Side-by-side layout */}
  <div className="flex flex-col sm:flex-row gap-6 items-center justify-center w-full max-w-6xl">
    {/* Reference video */}
    <div className="relative w-[320px] sm:w-[480px] h-[360px] sm:h-[480px]">
      <video
        ref={videoRef}
        src="/Video_Request_Single_Leg_Stand.mp4"
        autoPlay
        muted
        loop
        className="w-full h-full object-cover rounded"
      />
      <canvas
        ref={videoCanvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
    </div>

    {/* Webcam feed */}
    <div className="relative w-[320px] sm:w-[480px] h-[360px] sm:h-[480px]">
      <Webcam
        ref={webcamRef}
        mirrored
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={{ width: 640, height: 480 }}
        className="w-full h-full object-cover rounded"
      />
      <canvas
        ref={webcamCanvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
      {/* Status label */}
      <div className="absolute top-2 left-2 bg-white bg-opacity-80 px-3 py-1 rounded text-black font-semibold text-sm">
        {status === 'Ready' ? '✅ Ready' : '🔎 Detecting...'}
      </div>
      {/* Score overlay bottom center */}
      <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-80 text-white px-4 py-1 rounded text-lg font-bold">
        Score: {score !== null ? score.toFixed(2) : '...'}
      </div>
    </div>
  </div>

  {/* Score chart */}
  <div className="w-full max-w-4xl px-4 mt-6">
    <ScoreChart data={scoreHistory} />
  </div>
</div>
  );
};

export default RealTimeScoring;