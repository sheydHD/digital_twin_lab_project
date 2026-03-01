import React, { useState, useEffect } from 'react';
import { useSimulationViewModel } from '../viewmodels/useSimulationViewModel';
import './Dashboard.css';

// ── Stage label map ────────────────────────────────────────────────────────
const STAGE_LABELS: Record<string, string> = {
  idle: 'Ready',
  starting: 'Connecting\u2026',
  data_generation: 'Stage 1 \u2013 FEM Data Generation',
  calibration: 'Stage 2 \u2013 Bayesian Calibration (MCMC)',
  analysis: 'Stage 3 \u2013 Model Selection Analysis',
  done: 'Complete',
  error: 'Error',
};

// ── Helpers ────────────────────────────────────────────────────────────────
function plotLabel(url: string): string {
  const name = url.split('/').pop() ?? url;
  return name
    .replace('.png', '')
    .replace(/_/g, ' ')
    .replace(/Lh (\d+)/, '\u00a0L/h\u2009=\u2009$1')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function lhFromUrl(url: string): number | null {
  const m = url.match(/beam_comparison_Lh_(\d+(?:\.\d+)?)/);
  return m ? parseFloat(m[1]) : null;
}

// ── Main component ─────────────────────────────────────────────────────────
export const Dashboard: React.FC = () => {
  const { config, updateConfig, runSimulation, result, loading, error, progress } =
    useSimulationViewModel();

  // 3-phase state machine
  type Phase = 'config' | 'running' | 'results';
  const [phase, setPhase] = useState<Phase>('config');

  // Lightbox
  const [lightbox, setLightbox] = useState<string | null>(null);

  // Beam deflection slider
  const deflectionPlots = (result?.plots ?? [])
    .filter((u) => lhFromUrl(u) !== null)
    .sort((a, b) => (lhFromUrl(a) ?? 0) - (lhFromUrl(b) ?? 0));
  const [deflIdx, setDeflIdx] = useState(0);

  // Other plots (non-deflection)
  const otherPlots = (result?.plots ?? []).filter((u) => lhFromUrl(u) === null);

  // Transitions
  useEffect(() => {
    if (loading && !result) setPhase('running');
  }, [loading, result]);

  useEffect(() => {
    if (!loading && result) {
      setPhase('results');
      setDeflIdx(0);
    }
  }, [loading, result]);

  const handleRun = () => {
    runSimulation();
  };

  const handleReset = () => {
    setPhase('config');
  };

  const pct = progress.total > 0 ? Math.round((progress.step / progress.total) * 100) : 0;

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-title">Digital Twin Lab</span>
        <span className="app-subtitle">Bayesian Beam Model Selection</span>
        {phase === 'results' && (
          <button className="btn-ghost" onClick={handleReset}>
            ↺ New Simulation
          </button>
        )}
      </header>

      <main className="app-main">

        {/* ─── Phase 1: Config ─────────────────────────────────────── */}
        {phase === 'config' && (
          <div className="config-card">
            <h2 className="section-title">Simulation Configuration</h2>

            <div className="config-grid">
              <fieldset className="cfg-group">
                <legend>Beam Geometry</legend>
                <label>Length (m)
                  <input type="number" step="0.1" value={config.beam_parameters.length}
                    onChange={(e) => updateConfig('beam_parameters', 'length', parseFloat(e.target.value))} />
                </label>
                <label>Width (m)
                  <input type="number" step="0.01" value={config.beam_parameters.width}
                    onChange={(e) => updateConfig('beam_parameters', 'width', parseFloat(e.target.value))} />
                </label>
                <label>Aspect Ratios L/h (comma-separated)
                  <input type="text" value={config.beam_parameters.aspect_ratios.join(', ')}
                    onChange={(e) =>
                      updateConfig('beam_parameters', 'aspect_ratios',
                        e.target.value.split(',').map((v) => parseFloat(v.trim())).filter((n) => !isNaN(n)))
                    } />
                </label>
              </fieldset>

              <fieldset className="cfg-group">
                <legend>Material</legend>
                <label>Elastic Modulus (Pa)
                  <input type="number" value={config.material.elastic_modulus}
                    onChange={(e) => updateConfig('material', 'elastic_modulus', parseFloat(e.target.value))} />
                </label>
                <label>Poisson Ratio
                  <input type="number" step="0.01" min="0" max="0.5" value={config.material.poisson_ratio}
                    onChange={(e) => updateConfig('material', 'poisson_ratio', parseFloat(e.target.value))} />
                </label>
              </fieldset>

              <fieldset className="cfg-group">
                <legend>Bayesian Inference</legend>
                <label>MCMC Samples per chain
                  <input type="number" min="100" value={config.bayesian.n_samples}
                    onChange={(e) => updateConfig('bayesian', 'n_samples', parseInt(e.target.value))} />
                </label>
                <label>Warmup / Tune steps
                  <input type="number" min="100" value={config.bayesian.n_tune}
                    onChange={(e) => updateConfig('bayesian', 'n_tune', parseInt(e.target.value))} />
                </label>
                <label>Chains
                  <input type="number" min="1" max="8" value={config.bayesian.n_chains}
                    onChange={(e) => updateConfig('bayesian', 'n_chains', parseInt(e.target.value))} />
                </label>
              </fieldset>
            </div>

            <div className="config-actions">
              <button className="btn-primary" onClick={handleRun}>
                Run Simulation →
              </button>
            </div>
          </div>
        )}

        {/* ─── Phase 2: Running ────────────────────────────────────── */}
        {phase === 'running' && (
          <div className="running-card">
            <div className="running-header">
              <div className="loading-spinner"></div>
              <div className="running-title">{STAGE_LABELS[progress.stage] ?? progress.stage}</div>
            </div>

            <div className="prog-bar-track">
              <div className="prog-bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="prog-meta">
              {progress.total > 0
                ? `Step ${progress.step} of ${progress.total} · ${pct} %`
                : 'Initialising…'}
            </div>
            <p className="prog-message">{progress.message || ' '}</p>

            <div className="running-steps">
              {(['data_generation', 'calibration', 'analysis'] as const).map((s, i) => {
                const stageOrder = ['data_generation', 'calibration', 'analysis', 'done'];
                const curIdx = stageOrder.indexOf(progress.stage);
                const done = curIdx > i;
                const active = progress.stage === s;
                return (
                  <div key={s} className={`run-step${active ? ' active' : done ? ' done' : ''}`}>
                    <span className="run-step-icon">{done ? '✓' : active ? '●' : '○'}</span>
                    <span>{STAGE_LABELS[s]}</span>
                  </div>
                );
              })}
            </div>

            {error && (
              <div className="error-box">
                <strong>Error:</strong> {error}
                <button className="btn-ghost" onClick={handleReset} style={{ marginLeft: '1rem' }}>
                  Back to Config
                </button>
              </div>
            )}
          </div>
        )}

        {/* ─── Phase 3: Results ────────────────────────────────────── */}
        {phase === 'results' && result && (
          <div className="results-layout">

            {/* Summary row */}
            <div className="summary-row">
              <div className="summary-chip primary">
                <span className="chip-label">Recommendation</span>
                <span className="chip-value">{result.recommendedModel}</span>
              </div>
              {result.transitionPoint != null && (
                <div className="summary-chip">
                  <span className="chip-label">Transition L/h</span>
                  <span className="chip-value">{result.transitionPoint.toFixed(1)}</span>
                </div>
              )}
            </div>

            {/* Bayes factors table */}
            <div className="result-card">
              <h3>Log Bayes Factors</h3>
              <table className="bf-table">
                <thead>
                  <tr>
                    <th>L/h</th>
                    <th>Log Bayes Factor</th>
                    <th>Favours</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {result.logBayesFactors &&
                    Object.entries(result.logBayesFactors)
                      .sort(([a], [b]) => parseFloat(a) - parseFloat(b))
                      .map(([lh, bf]) => (
                        <tr key={lh}>
                          <td className="bf-lh">{lh}</td>
                          <td className={`bf-val ${bf > 0 ? 'eb' : 'timo'}`}>{bf.toFixed(4)}</td>
                          <td className={`bf-model ${bf > 0 ? 'eb' : 'timo'}`}>
                            {bf > 0 ? 'Euler-Bernoulli' : 'Timoshenko'}
                          </td>
                          <td className="bf-bar-cell">
                            <div className="bf-bar-track">
                              <div
                                className={`bf-bar ${bf > 0 ? 'eb' : 'timo'}`}
                                style={{ width: `${Math.min(Math.abs(bf) / 12 * 100, 100)}%` }}
                              />
                            </div>
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </div>

            {/* Beam deflection comparison slider */}
            {deflectionPlots.length > 0 && (
              <div className="result-card">
                <h3>Beam Deflection: EB vs Timoshenko</h3>
                <p className="card-hint">
                  Drag the slider to compare models across different aspect ratios.
                </p>
                <div className="deflection-slider-row">
                  <span className="slider-label">L/h</span>
                  <input
                    type="range"
                    min={0}
                    max={deflectionPlots.length - 1}
                    step={1}
                    value={deflIdx}
                    onChange={(e) => setDeflIdx(parseInt(e.target.value))}
                    className="deflection-slider"
                  />
                  <span className="slider-value">
                    {lhFromUrl(deflectionPlots[deflIdx]) ?? '—'}
                  </span>
                </div>
                <div className="deflection-ratio-labels">
                  {deflectionPlots.map((u, i) => (
                    <button
                      key={u}
                      className={`ratio-pip${i === deflIdx ? ' active' : ''}`}
                      onClick={() => setDeflIdx(i)}
                    >
                      {lhFromUrl(u)}
                    </button>
                  ))}
                </div>
                <img
                  src={deflectionPlots[deflIdx]}
                  alt={`Beam comparison L/h=${lhFromUrl(deflectionPlots[deflIdx])}`}
                  className="deflection-img"
                  onClick={() => setLightbox(deflectionPlots[deflIdx])}
                  title="Click to enlarge"
                />
              </div>
            )}

            {/* Other plots gallery */}
            {otherPlots.length > 0 && (
              <div className="result-card">
                <h3>Analysis Plots</h3>
                <p className="card-hint">Click any plot to enlarge.</p>
                <div className="thumb-grid">
                  {otherPlots.map((url) => (
                    <button key={url} className="thumb-btn" onClick={() => setLightbox(url)}>
                      <img src={url} alt={plotLabel(url)} className="thumb-img" />
                      <span className="thumb-caption">{plotLabel(url)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ─── Lightbox ───────────────────────────────────────────────── */}
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(null)}>
          <div className="lightbox-box" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightbox(null)}>×</button>
            <img src={lightbox} alt="enlarged plot" className="lightbox-img" />
            <p className="lightbox-caption">{plotLabel(lightbox)}</p>
          </div>
        </div>
      )}
    </div>
  );
};
