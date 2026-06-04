// Polling hook — calls the API every 2s until the job is done.
// Components just call useDetectionJob(jobId) and read {result, isPolling}.

import { useState, useEffect, useRef } from "react";
import { detectApi, DetectionResult } from "../lib/api";

export function useDetectionJob(jobId: string | null) {
  const [result,    setResult]    = useState<DetectionResult | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error,     setError]     = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!jobId) return;

    setIsPolling(true);
    setError(null);

    const poll = async () => {
      try {
        const { data } = await detectApi.getJob(jobId);
        setResult(data);
        if (data.status === "completed" || data.status === "failed") {
          setIsPolling(false);
          if (intervalRef.current) clearInterval(intervalRef.current);
        }
      } catch (err: any) {
        setError(err.response?.data?.detail ?? "Failed to fetch result");
        setIsPolling(false);
        if (intervalRef.current) clearInterval(intervalRef.current);
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [jobId]);

  return { result, isPolling, error };
}