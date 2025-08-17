import React from 'react'

const SizingCard = ({ sizing }) => {
  if (!sizing) return null

  return (
    <div className="card">
      <h3 className="card-title">Sizing</h3>
      
      {/* Main rows */}
      <div className="space-y-3">
        {sizing.x_star !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim text-sm">Target x*</span>
            <span className="big-number text-sm">
              {sizing.x_star.toFixed(3)}
            </span>
          </div>
        )}
        
        {sizing.current !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim text-sm">Current</span>
            <span className="big-number text-sm">
              {sizing.current.toFixed(3)}
            </span>
          </div>
        )}
        
        {sizing.kappa !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim text-sm">κ</span>
            <span className="big-number text-sm">
              {sizing.kappa.toFixed(3)}
            </span>
          </div>
        )}
      </div>

      {/* Sub-metrics */}
      <div className="mt-4 space-y-2 text-xs">
        {sizing.planned_step !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim">planned step</span>
            <span className="text-faint">{sizing.planned_step.toFixed(3)}</span>
          </div>
        )}
        
        {sizing.impact_eps_bp !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim">impact ε (bp)</span>
            <span className="text-faint">{sizing.impact_eps_bp.toFixed(1)}</span>
          </div>
        )}
        
        {sizing.last_calib_days !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim">last calib</span>
            <span className="text-faint">{sizing.last_calib_days}d</span>
          </div>
        )}
      </div>

      {/* Guard pills */}
      {sizing.guards && (
        <div className="mt-4 flex gap-2">
          {sizing.guards.funding && (
            <div className="pill pill-warning">
              funding guard: HALF
            </div>
          )}
          {sizing.guards.vol_drift && (
            <div className="pill pill-warning">
              vol-drift: CLEAR
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default SizingCard