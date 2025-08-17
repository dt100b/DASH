import React from 'react'

const GateCard = ({ gate }) => {
  if (!gate) return null

  const pRiskOn = gate.p_risk_on || 0
  const threshold = gate.threshold || 0.6
  const gateOpen = pRiskOn >= threshold

  return (
    <div className="card">
      <h3 className="card-title">Gate</h3>
      
      {/* Main percentage */}
      <div className="flex items-center justify-between mb-4">
        <div className="big-number">
          {(pRiskOn * 100).toFixed(1)}%
        </div>
        <div className={`pill ${gateOpen ? 'pill-long' : 'pill-flat'}`}>
          {gateOpen ? 'RISK-ON' : 'GATE-OFF'}
        </div>
      </div>

      {/* Sub-metrics */}
      <div className="space-y-2 text-xs">
        {gate.ess && (
          <div className="flex justify-between">
            <span className="text-dim">ESS</span>
            <span className="text-faint">{gate.ess.toFixed(2)}</span>
          </div>
        )}
        
        {gate.resample_spark && (
          <div className="flex justify-between">
            <span className="text-dim">resamples (7d):</span>
            <span className="spark">{gate.resample_spark}</span>
          </div>
        )}
        
        {gate.twist_status && (
          <div className="flex justify-between">
            <span className="text-dim">twist:</span>
            <span className="text-faint">{gate.twist_status}</span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-ink-1 text-xs text-dim">
        <div>threshold ≥ {threshold.toFixed(2)}</div>
        {gate.inertia_active && (
          <div className="mt-1">inertia rule applied</div>
        )}
      </div>
    </div>
  )
}

export default GateCard