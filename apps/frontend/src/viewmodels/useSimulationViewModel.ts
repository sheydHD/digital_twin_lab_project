import { useState, useEffect, useRef, useCallback } from 'react';
import type { SimulationConfig, SimulationProgress, SimulationResult } from '../models/types';

const DEFAULT_CONFIG: SimulationConfig = {
  beam_parameters: {
    length: 1.0,
    width: 0.1,
    aspect_ratios: [5, 8, 10, 12, 15, 19, 20, 30, 50, 60],
  },
  material: {
    elastic_modulus: 210.0e9,
    poisson_ratio: 0.3,
  },
  bayesian: {
    n_samples: 1500,
    n_tune: 800,
    n_chains: 4,
  },
  data: {
    noise_fraction: 0.0005,
  },
};

export const useSimulationViewModel = () => {
  const [config, setConfig] = useState<SimulationConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<SimulationProgress>({
    stage: 'idle', step: 0, total: 0, message: '', running: false,
  });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Use relative paths so Vite's dev-server proxy forwards them to the backend.
  const API_BASE = '';

  const fetchLatestResult = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/results/latest`);
      if (res.ok) {
        const data = await res.json();
        if (data) {
          setResult({
            jobId: data.jobId || 'unknown',
            status: 'completed',
            logBayesFactors: data.logBayesFactors,
            recommendedModel: data.recommendedModel,
            transitionPoint: data.transitionPoint,
            plots: data.plots || [],
          });
          return true;
        }
      }
    } catch (err) {
      console.error('Failed to fetch latest result:', err);
    }
    return false;
  }, []);

  // ── poll /api/progress every 4 s while a simulation is running ────────
  const startPolling = useCallback(() => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/progress`);
        if (res.ok) {
          const progData: SimulationProgress = await res.json();
          setProgress(progData);
          
          // If it just finished
          if (!progData.running && progData.stage === 'done') {
            stopPolling();
            setLoading(false);
            fetchLatestResult();
          } else if (!progData.running && progData.stage === 'error') {
            stopPolling();
            setLoading(false);
            setError(progData.message || 'Simulation failed on backend.');
          }
        }
      } catch { /* ignore transient errors */ }
    }, 4000);
  }, [fetchLatestResult]);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
        // 1. Fetch config
        const cfgRes = await fetch(`${API_BASE}/api/config`);
        if (cfgRes.ok) setConfig(await cfgRes.json());

        // 2. Check if already running
        const progRes = await fetch(`${API_BASE}/api/progress`);
        if (progRes.ok) {
          const progData: SimulationProgress = await progRes.json();
          if (progData.running) {
            setProgress(progData);
            setLoading(true);
            startPolling();
          } else if (progData.stage === 'done') {
            // Check if there's a result to show
            fetchLatestResult();
          }
        }
      } catch (err) {
        console.warn('Backend connection issue during init:', err);
      }
    };
    init();
    return stopPolling;
  }, [startPolling, stopPolling, fetchLatestResult]);

  const updateConfig = (section: keyof SimulationConfig, field: string, value: number | number[]) => {
    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value,
      },
    }));
  };

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress({ stage: 'starting', step: 0, total: 0, message: 'Connecting to backend…', running: true });
    
    try {
      // 1. Save config
      await fetch(`${API_BASE}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      // 2. Trigger simulation
      // Start polling FIRST so we catch the early progress
      startPolling();
      
      const res = await fetch(`${API_BASE}/api/simulate`, { method: 'POST' });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Simulation failed (${res.status}): ${detail}`);
      }
      
      // We don't setResult here anymore, startPolling handles it when it sees 'done'
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'An unexpected error occurred.';
      setError(msg);
      setLoading(false);
      stopPolling();
    }
  };

  return {
    config,
    updateConfig,
    runSimulation,
    result,
    loading,
    error,
    progress,
  };
};
