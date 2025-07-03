import React, { useEffect, useRef, useState } from 'react';
// import type { ReactWebcamProps } from 'react-webcam';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { scorePoseSimilarity } from '../utils/PoseScorer';
import referenceKeypoints from '../data/squat_demo_keypoints.json';
import Webcam from 'react-webcam';

/* const Webcam = dynamic(() => import('react-webcam'), {
  ssr: false,
}) as React.ComponentType<ReactWebcamProps>; */


const MODEL = poseDetection.SupportedModels.MoveNet;

const RealTimeScoring: React.FC = () => {
  const webcamRef = useRef<Webcam | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(null);

  useEffect(() => {
    const init = async () => {
      await poseDetection.createDetector(MODEL, {
        modelType: 'SinglePose.Lightning',
      }).then(setDetector);
    };

    init();
  }, []);

  useEffect(() => {
    if (!detector || !webcamRef.current?.video) return;

    const interval = setInterval(async () => {
      const video = webcamRef.current?.video;

      if (video && video.readyState === 4) {
        const poses = await detector.estimatePoses(video);
        if (poses.length > 0) {
          const liveKeypoints = poses[0].keypoints;

          // Match by current video time
          const frameIdx = Math.floor(video.currentTime * 5); // assuming 5fps extraction
          const refFrame = referenceKeypoints[frameIdx];

          if (refFrame) {
            const liveScore = scorePoseSimilarity(liveKeypoints, refFrame);
            setScore(liveScore);
          }
        }
      }
    }, 200);

    return () => clearInterval(interval);
  }, [detector]);

  return (
    <div className="flex flex-col items-center gap-4">
      <h2 className="text-xl font-semibold">Real-Time Movement Scoring</h2>

      <video
        src="/Video_Request_Single_Leg_Stand.mp4"
        autoPlay
        muted
        loop
        width={400}
      />

      <Webcam
        ref={webcamRef}
        audio={false}
        mirrored
        screenshotFormat="image/jpeg"
        videoConstraints={{ width: 640, height: 480 }}
      />

      <div className="mt-4 text-2xl font-bold">
        Score: {score !== null ? score.toFixed(2) : '...'}
      </div>
    </div>
  );
};

export default RealTimeScoring;
