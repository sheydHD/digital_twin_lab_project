import { renderHook, act } from '@testing-library/react';
import { useSimulationViewModel } from './useSimulationViewModel';
import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock the global fetch
global.fetch = vi.fn();

describe('useSimulationViewModel', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('initializes with the default config', () => {
    // Mock the fetch call for config to return a rejected promise so it falls back to default
    (global.fetch as any).mockRejectedValueOnce(new Error('Backend not available'));

    const { result } = renderHook(() => useSimulationViewModel());

    expect(result.current.config.beam_parameters.length).toBe(1.0);
    expect(result.current.config.material.elastic_modulus).toBe(210.0e9);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('updates configuration correctly', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('Backend not available'));
    const { result } = renderHook(() => useSimulationViewModel());

    act(() => {
      result.current.updateConfig('beam_parameters', 'length', 2.5);
    });

    expect(result.current.config.beam_parameters.length).toBe(2.5);
  });

  it('runs simulation successfully (mocked backend)', async () => {
    // 1st fetch: initial config load (mock failure to use default)
    (global.fetch as any).mockRejectedValueOnce(new Error('Backend not available'));
    
    const { result } = renderHook(() => useSimulationViewModel());

    // 2nd fetch: saving config POST
    (global.fetch as any).mockResolvedValueOnce({ ok: true });
    // 3rd fetch: run simulation POST
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        job_id: 'test-job-id',
        logBayesFactors: { '5': -10 },
        recommendedModel: 'Timoshenko',
        plots: ['plot1.png'],
      }),
    });

    await act(async () => {
      await result.current.runSimulation();
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.result).toEqual({
      jobId: 'test-job-id',
      status: 'completed',
      logBayesFactors: { '5': -10 },
      recommendedModel: 'Timoshenko',
      plots: ['plot1.png'],
    });
  });

  it('handles simulation errors gracefully', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('Backend not available'));
    const { result } = renderHook(() => useSimulationViewModel());

    // 2nd fetch: saving config POST fails
    (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

    // We need to use fake timers or wait because the ViewModel has a setTimeout for the mock demo
    vi.useFakeTimers();

    await act(async () => {
      await result.current.runSimulation();
    });

    act(() => {
      vi.runAllTimers();
    });

    expect(result.current.error).toBe('Network error');
    expect(result.current.loading).toBe(false);
    // Since it falls back to mock data on error in our implementation:
    expect(result.current.result?.jobId).toBe('demo-123');

    vi.useRealTimers();
  });
});
