import { render, screen, fireEvent } from '@testing-library/react';
import { Dashboard } from './Dashboard';
import { describe, it, expect, vi } from 'vitest';
import * as useSimulationViewModelModule from '../viewmodels/useSimulationViewModel';

// Mock the hook so we can control what the view receives
vi.mock('../viewmodels/useSimulationViewModel');

describe('Dashboard View', () => {
  const mockUpdateConfig = vi.fn();
  const mockRunSimulation = vi.fn();

  const defaultMockState = {
    config: {
      beam_parameters: { length: 1.0, width: 0.1, aspect_ratios: [5, 10] },
      material: { elastic_modulus: 210e9, poisson_ratio: 0.3 },
      bayesian: { n_samples: 800, n_tune: 400, n_chains: 2 },
      data: { noise_fraction: 0.0005 },
    },
    updateConfig: mockUpdateConfig,
    runSimulation: mockRunSimulation,
    result: null,
    loading: false,
    error: null,
  };

  it('renders the configuration form correctly', () => {
    vi.spyOn(useSimulationViewModelModule, 'useSimulationViewModel').mockReturnValue(defaultMockState);

    render(<Dashboard />);

    expect(screen.getByText('Digital Twin Lab - Bayesian Beam Model Selection')).toBeInTheDocument();
    expect(screen.getByLabelText(/Length/i)).toHaveValue(1.0);
    expect(screen.getByLabelText(/Elastic Modulus/i)).toHaveValue(210000000000);
  });

  it('calls updateConfig when an input is changed', () => {
    vi.spyOn(useSimulationViewModelModule, 'useSimulationViewModel').mockReturnValue(defaultMockState);

    render(<Dashboard />);

    const lengthInput = screen.getByLabelText(/Length/i);
    fireEvent.change(lengthInput, { target: { value: '2.5' } });

    expect(mockUpdateConfig).toHaveBeenCalledWith('beam_parameters', 'length', 2.5);
  });

  it('calls runSimulation when the button is clicked', () => {
    vi.spyOn(useSimulationViewModelModule, 'useSimulationViewModel').mockReturnValue(defaultMockState);

    render(<Dashboard />);

    const button = screen.getByRole('button', { name: /Run Simulation/i });
    fireEvent.click(button);

    expect(mockRunSimulation).toHaveBeenCalled();
  });

  it('displays loading state correctly', () => {
    vi.spyOn(useSimulationViewModelModule, 'useSimulationViewModel').mockReturnValue({
      ...defaultMockState,
      loading: true,
    });

    render(<Dashboard />);

    expect(screen.getByText('Processing bayesian inference...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Running Simulation.../i })).toBeDisabled();
  });

  it('displays results when available', () => {
    vi.spyOn(useSimulationViewModelModule, 'useSimulationViewModel').mockReturnValue({
      ...defaultMockState,
      result: {
        jobId: '123',
        status: 'completed',
        recommendedModel: 'Euler-Bernoulli',
        logBayesFactors: { '20': 0.42 },
        plots: [],
      },
    });

    render(<Dashboard />);

    expect(screen.getByText('Euler-Bernoulli')).toBeInTheDocument();
    expect(screen.getByText('0.420 (EB)')).toBeInTheDocument();
  });
});
