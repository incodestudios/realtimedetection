import React, { useEffect, useRef, useState } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { scorePoseByAngles } from '../utils/PoseScorer';
import { drawSkeleton } from '../utils/drawUtils';
import referenceKeypoints from '../data/squat_demo_keypoints.json';
import Webcam from 'react-webcam';
import ScoreChart from './ScoreChart';

const MODEL = poseDetection.SupportedModels.MoveNet;

const RealTimeScoring: React.FC = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [score, setScore] = useState<number | null>(null);
  const [scoreHistory, setScoreHistory] = useState<{ time: number; score: number }[]>([]);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);

  // Load pose detection model once
  useEffect(() => {
    const init = async () => {
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

        if (livePose?.keypoints && refKeypoints) {
          const liveScore = scorePoseByAngles(livePose.keypoints, refKeypoints);
          setScore(liveScore);
          setScoreHistory(prev => [...prev.slice(-50), {
            time: referenceVideo.currentTime,
            score: liveScore
          }]);

          const webcamCtx = webcamCanvasRef.current?.getContext('2d');
          if (webcamCtx) drawSkeleton(webcamCtx, livePose.keypoints);

          const videoCtx = videoCanvasRef.current?.getContext('2d');
          if (videoCtx) drawSkeleton(videoCtx, refKeypoints);
        }
      }, 200);

      // Cleanup loop
      return () => clearInterval(interval);
    };

    return () => clearInterval(waitForRefs);
  }, [detector]);

  return (
    <div className="flex flex-col items-center gap-6">
      <h2 className="text-xl font-semibold">Real-Time Movement Scoring</h2>

      {/* Reference video with skeleton overlay */}
      <div className="relative w-[640px] h-[480px]">
        <video
          ref={videoRef}
          src="/Video_Request_Single_Leg_Stand.mp4"
          autoPlay
          muted
          loop
          width={640}
          height={480}
          className="rounded"
        />
        <canvas
          ref={videoCanvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Webcam with skeleton overlay */}
      <div className="relative w-[640px] h-[480px]">
        <Webcam
          ref={webcamRef}
          mirrored
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 640, height: 480 }}
          className="rounded"
        />
        <canvas
          ref={webcamCanvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      <div className="text-2xl font-bold">
        Score: {score !== null ? score.toFixed(2) : '...'}
      </div>

      <div className="w-full max-w-3xl px-4">
        <ScoreChart data={scoreHistory} />
      </div>
    </div>
  );
};

export default RealTimeScoring;
