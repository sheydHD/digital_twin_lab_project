export interface BeamParameters {
  length: number;
  width: number;
  aspect_ratios: number[];
}

export interface MaterialProperties {
  elastic_modulus: number;
  poisson_ratio: number;
}

export interface BayesianParameters {
  n_samples: number;
  n_tune: number;
  n_chains: number;
}

export interface DataParameters {
  noise_fraction: number;
}

export interface SimulationConfig {
  beam_parameters: BeamParameters;
  material: MaterialProperties;
  bayesian: BayesianParameters;
  data: DataParameters;
}

export interface SimulationResult {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  logBayesFactors?: Record<string, number>;
  recommendedModel?: string;
  transitionPoint?: number;
  plots?: string[];
}

export interface SimulationProgress {
  stage: string;
  step: number;
  total: number;
  message: string;
  running: boolean;
}
