import { useState, useCallback } from "react";
import type {
  SystemMetrics,
  StreamStatus,
  SettingsState,
  PromptData,
} from "../types";
import { getDefaultDenoisingSteps, getDefaultResolution } from "../lib/utils";

export function useStreamState() {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpu: 0,
    gpu: 0,
    systemRAM: 0,
    vram: 0,
    fps: 0,
    latency: 0,
  });

  const [streamStatus, setStreamStatus] = useState<StreamStatus>({
    status: "Ready",
  });

  const [settings, setSettings] = useState<SettingsState>({
    pipelineId: "mycustom",
    resolution: getDefaultResolution("mycustom"), // Default resolution for mycustom
    seed: 42,
    denoisingSteps: getDefaultDenoisingSteps("mycustom"), // Default for mycustom
    manageCache: true, // Default manage cache
    paused: false, // Default to not paused (generating)
  });

  const [promptData, setPromptData] = useState<PromptData>({
    prompt: "",
    isProcessing: false,
  });

  const updateMetrics = useCallback((newMetrics: Partial<SystemMetrics>) => {
    setSystemMetrics(prev => ({ ...prev, ...newMetrics }));
  }, []);

  const updateStreamStatus = useCallback((newStatus: Partial<StreamStatus>) => {
    setStreamStatus(prev => ({ ...prev, ...newStatus }));
  }, []);

  const updateSettings = useCallback((newSettings: Partial<SettingsState>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const updatePrompt = useCallback((newPrompt: Partial<PromptData>) => {
    setPromptData(prev => ({ ...prev, ...newPrompt }));
  }, []);

  return {
    systemMetrics,
    streamStatus,
    settings,
    promptData,
    updateMetrics,
    updateStreamStatus,
    updateSettings,
    updatePrompt,
  };
}
