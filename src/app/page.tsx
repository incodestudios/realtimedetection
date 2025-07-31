'use client';

import RealTimeScoring from "@/components/RealTimeScoring";
import '@tensorflow/tfjs-backend-webgl';

export default function Home() {
  return (
    <div className="w-screen h-screen overflow-hidden">
      <RealTimeScoring />
    </div>
  );
}
