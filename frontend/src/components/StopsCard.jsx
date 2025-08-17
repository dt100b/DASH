import React from 'react'

const StopsCard = ({ stopping }) => {
  if (!stopping) return null

  const decision = stopping.decision || 'HOLD'

  return (
    <div className="card">
      <h3 className="card-title">Stops</h3>
      
      {/* Decision pill */}
      <div className="flex items-center justify-between mb-4">
        <div className={`pill ${decision === 'HOLD' ? 'pill-flat' : 'pill-warning'}`}>
          {decision}
        </div>
        {stopping.confidence !== undefined && (
          <div className="text-dim text-sm">
            conf {(stopping.confidence * 100).toFixed(0)}%
          </div>
        )}
      </div>

      {/* Sub-metrics */}
      <div className="space-y-2 text-xs">
        {stopping.bounds_low !== undefined && stopping.bounds_high !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim">bounds</span>
            <span className="text-faint">
              [{stopping.bounds_low.toFixed(2)}, {stopping.bounds_high.toFixed(2)}]
            </span>
          </div>
        )}
        
        {stopping.age_days !== undefined && stopping.max_days !== undefined && (
          <div className="flex justify-between">
            <span className="text-dim">age</span>
            <span className="text-faint">
              {stopping.age_days}/{stopping.max_days}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export default StopsCard