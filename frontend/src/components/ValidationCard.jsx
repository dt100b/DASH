import React from 'react'

const ValidationCard = ({ validation }) => {
  if (!validation) return null

  // Define validation criteria
  const checks = [
    {
      name: 'Hit',
      value: validation.hit_rate,
      threshold: 0.56,
      format: (v) => `${(v * 100).toFixed(0)}%`,
      operator: '>='
    },
    {
      name: 'Sharpe',
      value: validation.sharpe,
      threshold: 1.0,
      format: (v) => v.toFixed(1),
      operator: '>='
    },
    {
      name: 'MaxDD',
      value: validation.max_dd,
      threshold: 0.15,
      format: (v) => `${(v * 100).toFixed(0)}%`,
      operator: '<='
    },
    {
      name: 'Gate Off',
      value: validation.gate_off,
      threshold: 0.60,
      format: (v) => `${(v * 100).toFixed(0)}%`,
      operator: '<='
    }
  ]

  return (
    <div className="card">
      <h3 className="card-title">Validation</h3>
      
      {/* Ticks */}
      <ul className="ticks space-y-2">
        {checks.map((check, i) => {
          if (check.value === undefined) return null
          
          const passes = check.operator === '>=' 
            ? check.value >= check.threshold
            : check.value <= check.threshold

          return (
            <li key={i} className={passes ? 'ok' : 'no'}>
              <div className="flex justify-between items-center">
                <span className="text-dim text-xs">{check.name}≥{check.threshold}</span>
                <div className="flex items-center gap-2">
                  <span className="text-faint text-sm">{check.format(check.value)}</span>
                  <span className="text-xs">{passes ? '✓' : '—'}</span>
                </div>
              </div>
            </li>
          )
        })}
      </ul>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-ink-1 text-xs text-faint">
        {validation.slice_start && validation.slice_end && (
          <div>
            slice {validation.slice_start} → {validation.slice_end}
          </div>
        )}
        
        <div className="flex items-center gap-2 mt-1">
          {validation.costs_ok !== undefined && (
            <span>costs {validation.costs_ok ? '✓' : '—'}</span>
          )}
          {validation.embargo_ok !== undefined && (
            <span>embargo {validation.embargo_ok ? '✓' : '—'}</span>
          )}
          {validation.pass !== undefined && (
            <span className="font-medium">
              overall {validation.pass ? 'PASS' : 'FAIL'}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export default ValidationCard