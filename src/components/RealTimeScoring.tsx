// RealTimeScoring.tsx
//'use client';

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

const RealTimeScoring: React.FC = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [score, setScore] = useState<number | null>(null);
  const [scoreHistory, setScoreHistory] = useState<{ time: number; score: number }[]>([]);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);
  const [status, setStatus] = useState<'Detecting' | 'Ready'>('Detecting');

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

      if (webcamVideo?.readyState === 4 && referenceVideo?.readyState === 4) {
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
        const frameIdx = Math.floor(referenceVideo.currentTime * 5);
        const refKeypoints = referenceKeypoints[frameIdx];

        if (livePose?.keypoints) {
          const validKeypoints = livePose.keypoints.filter(
            (kp) => kp.score !== undefined && kp.score > 0.4
          );

          setStatus(validKeypoints.length > 5 ? 'Ready' : 'Detecting');

          if (refKeypoints) {
            const liveScore = scorePoseByAngles(livePose.keypoints, refKeypoints);
            setScore(liveScore);
            setScoreHistory((prev) => [
              ...prev.slice(-50),
              { time: referenceVideo.currentTime, score: liveScore },
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
    <div className="flex w-screen h-screen">
      {/* Left: Video */}
      <div className="relative flex-1 bg-black">
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
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Right: Webcam */}
      <div className="relative flex-1 bg-black">
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
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />

        {/* Detection status */}
        <div className="absolute top-2 left-2 bg-white bg-opacity-80 px-3 py-1 rounded text-black font-semibold text-sm">
          {status === 'Ready' ? '✅ Ready' : '🔎 Detecting...'}
        </div>

        {/* Score */}
        <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white px-3 py-1 rounded text-sm font-semibold">
          Score: {score !== null ? score.toFixed(2) : '...'}
        </div>
      </div>
      {/* Floating ScoreChart */}
      <div className="absolute bottom-4 left-4 w-[300px] h-[150px] bg-white bg-opacity-80 rounded p-2 z-50">
        <ScoreChart data={scoreHistory} />
      </div>
    </div>
  );
};

export default RealTimeScoring;