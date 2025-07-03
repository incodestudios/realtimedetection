// components/WebcamFeed.tsx
import React, { useRef, useEffect } from 'react';

const WebcamFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const setupWebcam = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    };

    setupWebcam();
  }, []);

  return (
    <div>
      <video
        ref={videoRef}
        id="webcam"
        autoPlay
        playsInline
        muted
        width={640}
        height={480}
        className="border"
      />
      <canvas id="pose-canvas" width={640} height={480} className="absolute top-0 left-0" />
    </div>
  );
};

export default WebcamFeed;
