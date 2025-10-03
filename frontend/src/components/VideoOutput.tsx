import { useEffect, useRef } from "react";
import { Card, CardContent } from "./ui/card";
import { Spinner } from "./ui/spinner";

interface VideoOutputProps {
  className?: string;
  remoteStream: MediaStream | null;
  isPipelineLoading?: boolean;
  isConnecting?: boolean;
  pipelineError?: string | null;
}

export function VideoOutput({
  className = "",
  remoteStream,
  isPipelineLoading = false,
  isConnecting = false,
  pipelineError = null,
}: VideoOutputProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardContent className="flex-1 flex items-center justify-center min-h-0">
        {remoteStream ? (
          <video
            ref={videoRef}
            className="max-w-full max-h-full object-contain"
            autoPlay
            muted
            playsInline
          />
        ) : pipelineError ? (
          <div className="text-center text-red-500 text-lg">
            <p>Pipeline Error</p>
            <p className="text-sm mt-2 max-w-md mx-auto">{pipelineError}</p>
          </div>
        ) : isPipelineLoading ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Loading pipeline...</p>
          </div>
        ) : isConnecting ? (
          <div className="text-center text-muted-foreground text-lg">
            <Spinner size={24} className="mx-auto mb-3" />
            <p>Connecting...</p>
          </div>
        ) : (
          <div className="text-center text-muted-foreground text-lg">
            Click "Start" when you are ready
          </div>
        )}
      </CardContent>
    </Card>
  );
}
