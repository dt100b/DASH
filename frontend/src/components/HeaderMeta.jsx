import React from 'react'

const HeaderMeta = ({ meta }) => {
  if (!meta) return null

  const staleness = meta.staleness_pct || 0
  const isStale = staleness > 0.5

  return (
    <div className="flex items-center gap-2 text-xs">
      {/* Staleness indicator */}
      <div className="flex items-center gap-1">
        <div className={`dot ${isStale ? 'dot-stale' : 'dot-fresh'}`} />
        <span className="text-faint">
          data @ {meta.snapshot_utc ? new Date(meta.snapshot_utc).toUTCString().slice(17, 25) : '--:--:--'}
        </span>
      </div>

      {/* Model versions */}
      {meta.models && (
        <div className="flex items-center gap-2">
          {meta.models.bias_version && (
            <span className="text-dim">bias {meta.models.bias_version}</span>
          )}
          {meta.models.stopping_version && (
            <span className="text-dim">stopping {meta.models.stopping_version}</span>
          )}
          {meta.models.sig_order && meta.models.sig_dims && (
            <span className="text-dim">
              sig k={meta.models.sig_order} d≈{meta.models.sig_dims}
            </span>
          )}
        </div>
      )}

      {/* Inertia status */}
      {meta.inertia_active && (
        <span className="text-dim">inertia: ON</span>
      )}
    </div>
  )
}

export default HeaderMeta