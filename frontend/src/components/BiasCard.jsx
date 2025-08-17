import React from 'react'

// Bias gauge component
const BiasGauge = ({ bias }) => {
  const normalizedBias = (bias + 1) / 2 // Convert -1..1 to 0..1
  const angle = normalizedBias * 180 - 90 // -90 to 90 degrees
  
  return (
    <div className="relative w-48 h-24">
      <svg className="w-48 h-24">
        {/* Arc background */}
        <path
          d="M 24 48 A 48 48 0 0 1 168 48"
          fill="none"
          stroke="var(--ink-1)"
          strokeWidth="2"
        />
        {/* Tick marks */}
        <line x1="24" y1="48" x2="24" y2="42" stroke="var(--txt-3)" strokeWidth="1" />
        <line x1="96" y1="6" x2="96" y2="12" stroke="var(--txt-3)" strokeWidth="1" />
        <line x1="168" y1="48" x2="168" y2="42" stroke="var(--txt-3)" strokeWidth="1" />
        {/* Needle */}
        <line
          x1="96"
          y1="48"
          x2={96 + 40 * Math.cos(angle * Math.PI / 180)}
          y2={48 + 40 * Math.sin(angle * Math.PI / 180)}
          stroke="var(--acc-1)"
          strokeWidth="2"
          className="transition-all duration-500"
        />
        <circle cx="96" cy="48" r="3" fill="var(--acc-1)" />
      </svg>
      <div className="absolute bottom-0 left-0 text-xs text-dim">SHORT</div>
      <div className="absolute bottom-0 right-0 text-xs text-dim">LONG</div>
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-xs text-dim">FLAT</div>
    </div>
  )
}

const BiasCard = ({ bias }) => {
  if (!bias) return null

  const biasValue = bias.value || 0

  return (
    <div className="card">
      <h3 className="card-title">Bias</h3>
      
      {/* Gauge and value */}
      <div className="flex flex-col items-center">
        <BiasGauge bias={biasValue} />
        <div className="big-number mt-4">
          {biasValue.toFixed(3)}
        </div>
      </div>

      {/* Top attributions as chips */}
      {bias.top_attribs && bias.top_attribs.length > 0 && (
        <div className="mt-4">
          <div className="text-dim text-xs mb-2">Top features:</div>
          <div className="flex flex-wrap gap-1">
            {bias.top_attribs.slice(0, 3).map((attr, i) => (
              <div key={i} className="chip">
                {attr.name}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Micro text footer */}
      <div className="mt-4 pt-3 border-t border-ink-1 text-xs text-faint">
        {bias.bema_alpha && (
          <span>BEMA α={bias.bema_alpha.toFixed(2)}</span>
        )}
        {bias.bema_alpha && bias.lr_regime && <span className="sep"> · </span>}
        {bias.lr_regime && (
          <span>LR: {bias.lr_regime}</span>
        )}
      </div>
    </div>
  )
}

export default BiasCard