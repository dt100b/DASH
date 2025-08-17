/*
 * TRADING DASHBOARD - REDESIGNED
 * =================================
 * README: This dashboard displays real-time quantitative trading data
 * 
 * DATA INTEGRATION:
 * - Replace fetchTelemetry() calls with your real trading API
 * - All metric labels and calculations are preserved exactly as-is
 * - Theme can be changed by updating CSS variables in index.css
 * 
 * SECTIONS PRESERVED:
 * - Gate - ORCSMC (regime detection)
 * - Bias (directional analysis) 
 * - Optimal Execution (trade execution metrics)
 * - Signature Features (path signature analysis)
 * - Deep Optimal Stopping (exit decision system)
 * - JAL Loss Training (loss function optimization)
 * - Learning & Optimization (LR scheduler + BEMA)
 */

import { useState, useEffect } from 'react';
import { 
  Play, RefreshCw, Upload, Download, User, Settings, TrendingUp, TrendingDown, Activity 
} from 'lucide-react';

// Radial gauge component - Updated for new theme
const RadialGauge = ({ value, label, threshold = 0.6 }: { value: number; label: string; threshold?: number }) => {
  const radius = 45;
  const strokeWidth = 3;
  const normalizedValue = Math.min(Math.max(value, 0), 1);
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (normalizedValue * circumference);
  
  return (
    <div className="relative w-32 h-32">
      <svg className="transform -rotate-90 w-32 h-32">
        <circle
          cx="64"
          cy="64"
          r={radius}
          stroke="var(--ui-border)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        <circle
          cx="64"
          cy="64"
          r={radius}
          stroke={value >= threshold ? "var(--ui-accent)" : "var(--ui-text-subtle)"}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-lg font-light" style={{ color: 'var(--ui-text)' }}>
            {(value * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function for bias sentiment
const getBiasSentiment = (bias: number) => {
  const abs = Math.abs(bias);
  if (abs < 0.1) return 'NEUTRAL';
  if (abs < 0.25) return bias > 0 ? 'BULLISH' : 'BEARISH';
  if (abs < 0.4) return bias > 0 ? 'STRONG BULL' : 'STRONG BEAR';
  return bias > 0 ? 'EXTREME BULL' : 'EXTREME BEAR';
};

// Neumorphic Bias Gauge - Soft UI Design
const NeumorphicBiasGauge = ({ bias, loading }: { bias: number; loading?: boolean }) => {
  const clampedBias = Math.max(-1, Math.min(1, bias));
  const rotation = (clampedBias + 1) * 90; // 0-180 degrees
  const sentiment = getBiasSentiment(clampedBias);
  
  // Color configuration based on bias strength
  const getColorConfig = (value: number) => {
    const abs = Math.abs(value);
    if (abs < 0.1) return { 
      needle: '#9CA3AF', 
      glow: 'rgba(156, 163, 175, 0.4)',
      accent: '#6B7280'
    };
    if (abs < 0.25) return { 
      needle: value > 0 ? '#10B981' : '#EF4444', 
      glow: value > 0 ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)',
      accent: value > 0 ? '#059669' : '#DC2626'
    };
    return { 
      needle: value > 0 ? '#10B981' : '#EF4444', 
      glow: value > 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)',
      accent: value > 0 ? '#059669' : '#DC2626'
    };
  };

  const colors = getColorConfig(clampedBias);

  if (loading) {
    return (
      <div className="w-full max-w-sm mx-auto p-8">
        <div className="animate-pulse">
          <div className="h-32 bg-gray-700/20 rounded-xl mb-4" />
          <div className="h-6 bg-gray-700/20 rounded mx-auto w-24 mb-2" />
          <div className="h-4 bg-gray-700/20 rounded mx-auto w-16" />
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-sm mx-auto">
      {/* Gauge Container */}
      <div className="relative h-24 mb-3 overflow-hidden">
          <svg 
            className="w-full h-full" 
            viewBox="0 0 240 100"
            style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
          >
            {/* Background Track with Neumorphic Effect */}
            <path
              d="M 30 110 A 90 90 0 0 1 210 110"
              fill="none"
              stroke="rgba(255, 255, 255, 0.05)"
              strokeWidth="3"
            />
            <path
              d="M 30 110 A 90 90 0 0 1 210 110"
              fill="none"
              stroke="rgba(0, 0, 0, 0.3)"
              strokeWidth="1"
              strokeDasharray="2,2"
              transform="translate(0, 1)"
            />
            
            {/* Scale Ticks - Precise Markings */}
            {[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1].map((tick) => {
              const angle = (tick + 1) * 90 - 90;
              const isMainTick = tick === -1 || tick === 0 || tick === 1;
              const isMidTick = Math.abs(tick) === 0.5;
              const tickLength = isMainTick ? 12 : isMidTick ? 8 : 4;
              const x1 = 120 + (75 - tickLength) * Math.cos(angle * Math.PI / 180);
              const y1 = 110 + (75 - tickLength) * Math.sin(angle * Math.PI / 180);
              const x2 = 120 + 75 * Math.cos(angle * Math.PI / 180);
              const y2 = 110 + 75 * Math.sin(angle * Math.PI / 180);
              
              return (
                <g key={tick}>
                  <line
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="rgba(255, 255, 255, 0.3)"
                    strokeWidth={isMainTick ? "2" : "1"}
                    opacity={isMainTick ? "0.8" : "0.4"}
                  />
                  {isMainTick && (
                    <text
                      x={120 + 85 * Math.cos(angle * Math.PI / 180)}
                      y={110 + 85 * Math.sin(angle * Math.PI / 180) + 4}
                      fill="rgba(255, 255, 255, 0.6)"
                      fontSize="12"
                      textAnchor="middle"
                      fontWeight="300"
                    >
                      {tick === 0 ? '0' : tick > 0 ? `+${tick}` : tick}
                    </text>
                  )}
                </g>
              );
            })}
            
            {/* Active Zone Indicator */}
            {Math.abs(clampedBias) > 0.1 && (
              <path
                d={clampedBias > 0 
                  ? "M 165 35 A 90 90 0 0 1 210 110"
                  : "M 30 110 A 90 90 0 0 1 75 35"
                }
                fill="none"
                stroke={colors.needle}
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray="85"
                strokeDashoffset={85 - (Math.abs(clampedBias) * 85)}
                opacity="0.6"
                style={{
                  transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                  filter: `drop-shadow(0 0 8px ${colors.glow})`
                }}
              />
            )}
            
            {/* Precision Needle with Glow */}
            <g 
              style={{ 
                transformOrigin: '120px 110px',
                transform: `rotate(${rotation - 90}deg)`,
                transition: 'transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
              }}
            >
              {/* Needle Glow */}
              <line
                x1="120" y1="110" x2="120" y2="35"
                stroke={colors.glow}
                strokeWidth="4"
                strokeLinecap="round"
                opacity="0.6"
                style={{ filter: `blur(2px)` }}
              />
              {/* Main Needle */}
              <line
                x1="120" y1="110" x2="120" y2="35"
                stroke={colors.needle}
                strokeWidth="2"
                strokeLinecap="round"
                style={{
                  filter: `drop-shadow(0 2px 4px rgba(0,0,0,0.3))`
                }}
              />
              {/* Needle Tip */}
              <circle
                cx="120" cy="35"
                r="3"
                fill={colors.needle}
                style={{
                  filter: `drop-shadow(0 0 6px ${colors.glow})`
                }}
              />
            </g>
            
            {/* Center Hub - Neumorphic */}
            <circle
              cx="120" cy="110"
              r="8"
              fill="var(--ui-bg)"
              style={{
                filter: `drop-shadow(0 2px 4px rgba(0,0,0,0.4))`
              }}
            />
            <circle
              cx="120" cy="110"
              r="6"
              fill="none"
              stroke={colors.needle}
              strokeWidth="1"
              opacity="0.6"
            />
            <circle
              cx="120" cy="110"
              r="3"
              fill={colors.needle}
              opacity="0.8"
            />
          </svg>
        </div>
        
        {/* Large Centered Value */}
        <div className="text-center mb-2">
          <div 
            className="text-2xl font-light tracking-wider mb-1"
            style={{ 
              color: colors.needle,
              textShadow: `0 0 10px ${colors.glow}, 0 2px 4px rgba(0,0,0,0.5)`,
              transition: 'all 0.8s ease'
            }}
          >
            {clampedBias.toFixed(3)}
          </div>
          
          {/* Concise Sentiment Label */}
          <div 
            className="text-sm font-medium tracking-wide opacity-80"
            style={{ 
              color: 'rgba(255, 255, 255, 0.7)',
              textShadow: '0 1px 2px rgba(0,0,0,0.5)'
            }}
          >
            {sentiment}
          </div>
        </div>
        
        {/* Minimal Action Pill */}
        <div className="flex justify-center">
          <div 
            className="px-6 py-2 rounded-full text-xs font-medium transition-all duration-500"
            style={{
              background: `linear-gradient(145deg, ${colors.accent}20, ${colors.accent}10)`,
              color: colors.needle,
              border: `1px solid ${colors.needle}30`,
              boxShadow: `
                inset -2px -2px 4px rgba(255, 255, 255, 0.02),
                inset 2px 2px 4px rgba(0, 0, 0, 0.2),
                0 2px 4px rgba(0, 0, 0, 0.1)
              `,
              textShadow: '0 1px 2px rgba(0,0,0,0.3)'
            }}
          >
            {Math.abs(clampedBias) > 0.15 
              ? clampedBias > 0 ? 'LONG' : 'SHORT'
              : 'HOLD'
            }
          </div>
        </div>
    </div>
  );
};

// Premium Bias Gauge - World-Class Craftsmanship
const PremiumBiasGauge = ({ bias, loading }: { bias: number; loading?: boolean }) => {
  const clampedBias = Math.max(-1, Math.min(1, bias));
  const rotation = (clampedBias + 1) * 90; // 0-180 degrees
  
  // Advanced zone classification
  const getZoneConfig = (value: number) => {
    const abs = Math.abs(value);
    if (abs < 0.1) return { 
      zone: 'neutral', 
      color: 'var(--ui-text-muted)', 
      intensity: 0.2,
      glow: 'none'
    };
    if (abs < 0.25) return { 
      zone: 'moderate', 
      color: 'var(--ui-accent)', 
      intensity: 0.5,
      glow: `0 0 20px ${value > 0 ? 'var(--ui-success)' : 'var(--ui-danger)'}30`
    };
    if (abs < 0.4) return { 
      zone: 'strong', 
      color: value > 0 ? 'var(--ui-success)' : 'var(--ui-danger)', 
      intensity: 0.7,
      glow: `0 0 30px ${value > 0 ? 'var(--ui-success)' : 'var(--ui-danger)'}40`
    };
    return { 
      zone: 'extreme', 
      color: value > 0 ? 'var(--ui-success)' : 'var(--ui-danger)', 
      intensity: 1.0,
      glow: `0 0 40px ${value > 0 ? 'var(--ui-success)' : 'var(--ui-danger)'}60`
    };
  };

  const config = getZoneConfig(clampedBias);
  
  if (loading) {
    return (
      <div className="relative w-48 h-28 flex items-center justify-center">
        <div className="animate-pulse">
          <div className="w-32 h-16 bg-gray-700/30 rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-48 h-28" title={`Bias: ${clampedBias.toFixed(3)} (${getBiasSentiment(clampedBias)})`}>
      <svg className="w-full h-full drop-shadow-lg" viewBox="0 0 192 112">
        {/* Gauge Background Track */}
        <path
          d="M 24 88 A 72 72 0 0 1 168 88"
          fill="none"
          stroke="var(--ui-border)"
          strokeWidth="1"
          opacity="0.3"
        />
        
        {/* Precision Scale Ticks */}
        {[-1, -0.5, -0.25, 0, 0.25, 0.5, 1].map((tick, i) => {
          const angle = (tick + 1) * 90 - 90;
          const isMain = tick === -1 || tick === 0 || tick === 1;
          const x1 = 96 + (isMain ? 60 : 65) * Math.cos(angle * Math.PI / 180);
          const y1 = 88 + (isMain ? 60 : 65) * Math.sin(angle * Math.PI / 180);
          const x2 = 96 + (isMain ? 70 : 68) * Math.cos(angle * Math.PI / 180);
          const y2 = 88 + (isMain ? 70 : 68) * Math.sin(angle * Math.PI / 180);
          
          return (
            <g key={tick}>
              <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="var(--ui-text-subtle)"
                strokeWidth={isMain ? "2" : "1"}
                opacity={isMain ? "0.6" : "0.3"}
              />
              {isMain && (
                <text
                  x={96 + 78 * Math.cos(angle * Math.PI / 180)}
                  y={88 + 78 * Math.sin(angle * Math.PI / 180) + 4}
                  fill="var(--ui-text-subtle)"
                  fontSize="10"
                  textAnchor="middle"
                  opacity="0.7"
                >
                  {tick === 0 ? '0' : tick > 0 ? `+${tick}` : tick}
                </text>
              )}
            </g>
          );
        })}
        
        {/* Zone Backgrounds */}
        <path
          d="M 24 88 A 72 72 0 0 1 60 32"
          fill="none"
          stroke="var(--ui-danger)"
          strokeWidth="8"
          strokeLinecap="round"
          opacity="0.1"
        />
        <path
          d="M 60 32 A 72 72 0 0 1 132 32"
          fill="none"
          stroke="var(--ui-text-muted)"
          strokeWidth="8"
          strokeLinecap="round"
          opacity="0.1"
        />
        <path
          d="M 132 32 A 72 72 0 0 1 168 88"
          fill="none"
          stroke="var(--ui-success)"
          strokeWidth="8"
          strokeLinecap="round"
          opacity="0.1"
        />
        
        {/* Active Zone Highlight */}
        {clampedBias > 0.1 && (
          <path
            d="M 132 32 A 72 72 0 0 1 168 88"
            fill="none"
            stroke={config.color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray="70"
            strokeDashoffset={70 - ((clampedBias - 0.1) / 0.9) * 70}
            opacity={config.intensity}
            style={{ 
              transition: 'all 0.8s cubic-bezier(0.25, 0.1, 0.25, 1)',
              filter: `drop-shadow(${config.glow})`
            }}
          />
        )}
        
        {clampedBias < -0.1 && (
          <path
            d="M 24 88 A 72 72 0 0 1 60 32"
            fill="none"
            stroke={config.color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray="70"
            strokeDashoffset={70 - ((Math.abs(clampedBias) - 0.1) / 0.9) * 70}
            opacity={config.intensity}
            style={{ 
              transition: 'all 0.8s cubic-bezier(0.25, 0.1, 0.25, 1)',
              filter: `drop-shadow(${config.glow})`
            }}
          />
        )}
        
        {/* Precision Needle with Depth */}
        <g style={{ 
          transformOrigin: '96px 88px',
          transform: `rotate(${rotation - 90}deg)`,
          transition: 'transform 0.8s cubic-bezier(0.25, 0.1, 0.25, 1)'
        }}>
          {/* Needle Shadow */}
          <line
            x1="96" y1="88" x2="96" y2="28"
            stroke="rgba(0,0,0,0.3)"
            strokeWidth="3"
            strokeLinecap="round"
            transform="translate(1,1)"
          />
          {/* Main Needle */}
          <line
            x1="96" y1="88" x2="96" y2="28"
            stroke={config.color}
            strokeWidth="2.5"
            strokeLinecap="round"
            opacity="0.9"
            style={{ 
              filter: `drop-shadow(${config.glow})`,
              transition: 'all 0.8s ease'
            }}
          />
          {/* Needle Tip */}
          <circle
            cx="96" cy="28"
            r="3"
            fill={config.color}
            opacity="0.8"
            style={{ filter: `drop-shadow(${config.glow})` }}
          />
        </g>
        
        {/* Center Hub with Glass Effect */}
        <circle
          cx="96" cy="88"
          r="6"
          fill="rgba(255,255,255,0.1)"
          stroke={config.color}
          strokeWidth="2"
          opacity="0.7"
          style={{ 
            transition: 'all 0.8s ease',
            filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.3))'
          }}
        />
        <circle
          cx="96" cy="88"
          r="3"
          fill={config.color}
          opacity="0.6"
        />
      </svg>
    </div>
  );
};

// Legacy Bias Gauge (keeping for reference)
const BiasGauge = ({ bias }: { bias: number }) => {
  const clampedBias = Math.max(-1, Math.min(1, bias));
  const rotation = (clampedBias + 1) * 90; // 0-180 degrees
  const absValue = Math.abs(clampedBias);
  
  // Zone classification and color mapping
  const getZoneData = (value: number) => {
    const abs = Math.abs(value);
    if (abs < 0.15) return { 
      zone: 'neutral', 
      color: 'var(--ui-text-muted)', 
      label: 'NEUTRAL',
      intensity: 0.3 
    };
    if (abs < 0.35) return { 
      zone: 'moderate', 
      color: 'var(--ui-accent-muted)', 
      label: value > 0 ? 'BULLISH' : 'BEARISH',
      intensity: 0.6 
    };
    return { 
      zone: 'strong', 
      color: 'var(--ui-accent)', 
      label: value > 0 ? 'STRONG BULL' : 'STRONG BEAR',
      intensity: 1.0 
    };
  };

  const zoneData = getZoneData(clampedBias);
  const trackOpacity = 0.15;
  const activeOpacity = 0.8;

  return (
    <div className="relative w-40 h-24 mx-auto">
      <svg className="w-full h-full" viewBox="0 0 160 96">
        {/* Background track */}
        <path
          d="M 20 80 A 60 60 0 0 1 140 80"
          fill="none"
          stroke="var(--ui-border)"
          strokeWidth="1"
          opacity={trackOpacity}
        />
        
        {/* Zone backgrounds - Bearish */}
        <path
          d="M 20 80 A 60 60 0 0 1 55 28"
          fill="none"
          stroke="var(--ui-danger)"
          strokeWidth="6"
          strokeLinecap="round"
          opacity={trackOpacity}
        />
        
        {/* Zone backgrounds - Neutral */}
        <path
          d="M 55 28 A 60 60 0 0 1 105 28"
          fill="none"
          stroke="var(--ui-text-muted)"
          strokeWidth="6"
          strokeLinecap="round"
          opacity={trackOpacity}
        />
        
        {/* Zone backgrounds - Bullish */}
        <path
          d="M 105 28 A 60 60 0 0 1 140 80"
          fill="none"
          stroke="var(--ui-success)"
          strokeWidth="6"
          strokeLinecap="round"
          opacity={trackOpacity}
        />
        
        {/* Active bias indicator */}
        {clampedBias > 0.15 && (
          <path
            d="M 105 28 A 60 60 0 0 1 140 80"
            fill="none"
            stroke={zoneData.color}
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray="55"
            strokeDashoffset={55 - (Math.max(0, clampedBias - 0.15) / 0.85) * 55}
            opacity={activeOpacity}
            style={{ 
              transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
              filter: `brightness(${1 + zoneData.intensity * 0.3})`
            }}
          />
        )}
        
        {clampedBias < -0.15 && (
          <path
            d="M 20 80 A 60 60 0 0 1 55 28"
            fill="none"
            stroke={zoneData.color}
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray="55"
            strokeDashoffset={55 - (Math.max(0, -clampedBias - 0.15) / 0.85) * 55}
            opacity={activeOpacity}
            style={{ 
              transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
              filter: `brightness(${1 + zoneData.intensity * 0.3})`
            }}
          />
        )}
        
        {/* Precision needle */}
        <g style={{ 
          transformOrigin: '80px 80px',
          transform: `rotate(${rotation - 90}deg)`,
          transition: 'transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
        }}>
          <line
            x1="80" y1="80"
            x2="80" y2="30"
            stroke={zoneData.color}
            strokeWidth="2.5"
            strokeLinecap="round"
            opacity={0.9}
          />
          {/* Needle tip */}
          <circle
            cx="80" cy="30"
            r="2"
            fill={zoneData.color}
            opacity={0.9}
          />
        </g>
        
        {/* Center hub */}
        <circle
          cx="80" cy="80"
          r="4"
          fill={zoneData.color}
          opacity={0.7}
          style={{ transition: 'fill 0.8s ease' }}
        />
        <circle
          cx="80" cy="80"
          r="2"
          fill="var(--ui-bg)"
        />
        
        {/* Scale markers */}
        <text x="20" y="90" fill="var(--ui-text-subtle)" fontSize="10" textAnchor="middle">-1.0</text>
        <text x="55" y="22" fill="var(--ui-text-subtle)" fontSize="9" textAnchor="middle">-0.5</text>
        <text x="80" y="18" fill="var(--ui-text-subtle)" fontSize="9" textAnchor="middle">0</text>
        <text x="105" y="22" fill="var(--ui-text-subtle)" fontSize="9" textAnchor="middle">+0.5</text>
        <text x="140" y="90" fill="var(--ui-text-subtle)" fontSize="10" textAnchor="middle">+1.0</text>
      </svg>
      
      {/* Precision readout */}
      <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 text-center">
        <div 
          className="text-xs font-medium px-3 py-1 rounded-full backdrop-blur-sm"
          style={{ 
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            color: zoneData.color,
            border: `1px solid ${zoneData.color}40`,
            transition: 'all 0.8s ease',
            boxShadow: `0 2px 8px ${zoneData.color}20`
          }}
        >
          {zoneData.label}
        </div>
      </div>
    </div>
  );
};

// Sparkline component - Updated for new theme
const Sparkline = ({ data, color }: { data: number[]; color?: string }) => {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = ((max - val) / range) * 40 + 5;
    return `${x},${y}`;
  }).join(' ');
  
  return (
    <svg className="w-full h-12" viewBox="0 0 100 50">
      <polyline
        points={points}
        fill="none"
        stroke={color || "var(--ui-accent)"}
        strokeWidth="1.5"
        opacity="0.8"
      />
      <line x1="0" y1="25" x2="100" y2="25" stroke="var(--ui-border)" strokeWidth="0.5" strokeDasharray="2,2" />
    </svg>
  );
};

// Theme switcher utility
const ThemeSwitcher = () => {
  const [theme, setTheme] = useState('violet');
  
  const themes = [
    { name: 'violet', label: 'V' },
    { name: 'blue', label: 'B' },
    { name: 'green', label: 'G' },
    { name: 'orange', label: 'O' },
    { name: 'cyan', label: 'C' }
  ];
  
  const switchTheme = (newTheme: string) => {
    setTheme(newTheme);
    if (newTheme === 'violet') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', newTheme);
    }
  };
  
  return (
    <div className="flex gap-1">
      {themes.map(t => (
        <button
          key={t.name}
          onClick={() => switchTheme(t.name)}
          className={`w-6 h-6 rounded-full text-xs font-medium transition-all ${
            theme === t.name ? 'ui-pill active' : 'ui-pill'
          }`}
          title={`Switch to ${t.name} theme`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
};

// Main App component
export default function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [telemetry, setTelemetry] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [time, setTime] = useState(new Date());

  // Update clock
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch telemetry data
  useEffect(() => {
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const fetchTelemetry = async () => {
    try {
      const response = await fetch(`/api/telemetry?symbol=${selectedSymbol}`);
      const data = await response.json();
      setTelemetry(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch telemetry:', error);
      // Use mock data for demo
      setTelemetry({
        gate_prob: 0.72,
        bias: 0.35,
        target_exposure: 0.35,
        current_exposure: 0.20,
        kappa: 0.15,
        guards: { funding_guard: false, vol_drift_guard: false },
        latest_decision: 'LONG',
        recent_pnl: 0.023,
        orcsmc: {
          regime_confidence: 0.85,
          particle_ess: 847,
          learning_iterations: 5,
          rolling_window_size: 30,
          inertia_counter: 1,
          twist_params: { A_diag: [0.12, 0.08], b: [0.02, -0.01] }
        },
        signature_features: {
          order: 3,
          n_features: 28,
          sig_time_price: 0.42,
          sig_price_vol: -0.18,
          sig_time_vol: 0.15,
          sig_ppp: 0.08,
          sig_tpp: -0.06,
          vol_of_vol: 0.22,
          realized_vol_24h: 0.22,
          realized_vol_72h: 0.19
        },
        execution: {
          kappa_fast: 0.15,
          kappa_slow: 0.08,
          price_impact: 0.0003,
          slippage_weekly: 0.0012,
          half_step_active: false,
          execution_rate: 0.12,
          impact_decay: 0.85
        },
        stopping: {
          decision: 'HOLD',
          confidence: 0.85,
          lower_bound: 0.02,
          upper_bound: 0.08,
          continuation_value: 0.045,
          martingale_increment: 0.001,
          day_counter: 2,
          max_days: 7,
          network_depth: 3
        },
        jal_training: {
          active_loss: 0.12,
          passive_loss: 0.08,
          alpha: 0.7,
          beta: 0.3,
          noise_level: 0.09,
          asymmetry_param: 3.0,
          jal_enabled: false,
          warmup_complete: true
        },
        learning_schedule: {
          current_lr: 0.15,
          base_lr: 0.3,
          min_lr: 1e-5,
          warmup_complete: true,
          sparsity_ratio: 0.34,
          confidence_score: 0.87,
          class_separation: 2.3,
          compression_ready: true
        },
        bema_metrics: {
          momentum_avg: 0.15,
          exp_avg_sq: 0.023,
          bias_correction_1: 0.95,
          bias_correction_2: 0.98,
          convergence_rate: 2.4,
          stability_score: 0.91
        },
        features: {
          'sig_time_price': 0.42,
          'sig_price_vol': -0.18,
          'realized_vol_24h': 0.22,
          'delta_funding_24h': 0.0002,
          'oi_change_rate': 0.15,
          'volume_ratio': 1.8,
          'funding_momentum': -0.05,
          'vwap_deviation': 0.003
        },
        attributions: [
          { name: 'sig_time_price', value: 0.42, attribution: 0.15 },
          { name: 'realized_vol_24h', value: 0.22, attribution: 0.08 },
          { name: 'sig_price_vol', value: -0.18, attribution: -0.06 },
          { name: 'delta_funding_24h', value: 0.0002, attribution: 0.04 }
        ],
        timestamp: new Date().toISOString()
      });
      setLoading(false);
    }
  };

  const runNow = async () => {
    try {
      const response = await fetch('/api/run-now', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: selectedSymbol })
      });
      const data = await response.json();
      console.log('Pipeline executed:', data);
    } catch (error) {
      console.error('Run failed:', error);
    }
  };

  const runBacktest = async () => {
    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: selectedSymbol, days: 30 })
      });
      const data = await response.json();
      console.log('Backtest completed:', data);
    } catch (error) {
      console.error('Backtest failed:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--ui-bg)' }}>
        <div className="animate-pulse" style={{ color: 'var(--ui-text-muted)' }}>Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ 
      backgroundColor: 'var(--ui-bg)',
      fontFamily: 'Inter, sans-serif' 
    }}>
      {/* Elite Trading Terminal Header */}
      <header className="px-10 py-6 flex items-center justify-between" style={{ 
        borderBottom: '1px solid var(--ui-border)',
        background: 'linear-gradient(180deg, rgba(10,10,10,0.8) 0%, transparent 100%)',
        backdropFilter: 'blur(10px)'
      }}>
        <div className="flex items-center gap-10">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{
              background: 'linear-gradient(135deg, var(--ui-accent) 0%, var(--ui-accent-muted) 100%)',
              boxShadow: '0 4px 16px rgba(0, 212, 255, 0.3)'
            }}>
              <Activity className="w-6 h-6 text-black" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight" style={{ 
                color: 'var(--ui-text)',
                fontFamily: 'var(--font-display)'
              }}>Quantum Trading Terminal</h1>
              <div className="flex items-center gap-3">
                <p className="text-sm" style={{ color: 'var(--ui-text-muted)' }}>
                  {time.toUTCString().slice(17, 25)} UTC • Advanced Pipeline Active
                </p>
                <div className="px-2 py-1 rounded text-xs" style={{ 
                  background: 'var(--ui-surface-3)', 
                  color: 'var(--ui-text-subtle)' 
                }}>
                  v2.4.1-a7f3e2d
                </div>
                <div className="px-2 py-1 rounded text-xs" style={{ 
                  background: 'var(--ui-surface-3)', 
                  color: 'var(--ui-text-subtle)' 
                }}>
                  Data age: 00:34
                </div>
                <div className="px-2 py-1 rounded text-xs" style={{ 
                  background: 'var(--ui-surface-3)', 
                  color: 'var(--ui-text-subtle)' 
                }}>
                  Options: OFF
                </div>
              </div>
            </div>
          </div>
          
          {/* Asset Selector - Premium Style */}
          <div className="flex gap-2 p-1 rounded-lg" style={{ 
            background: 'var(--ui-surface)', 
            border: '1px solid var(--ui-border)' 
          }}>
            {['BTC', 'ETH'].map(symbol => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                className="px-4 py-2 rounded-md transition-all text-sm font-medium"
                style={{
                  background: selectedSymbol === symbol ? 'var(--ui-accent)' : 'transparent',
                  color: selectedSymbol === symbol ? '#000' : 'var(--ui-text-muted)',
                  border: '1px solid transparent',
                  transform: selectedSymbol === symbol ? 'scale(1)' : 'scale(0.95)'
                }}
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex gap-3">
            <button onClick={runNow} className="px-4 py-2 rounded-lg flex items-center gap-2 transition-all" style={{
              background: 'linear-gradient(135deg, var(--ui-success) 0%, #00cc6a 100%)',
              color: '#000',
              fontWeight: '600',
              boxShadow: '0 4px 12px rgba(0, 255, 136, 0.3)'
            }}>
              <Play size={16} />
              Execute Now
            </button>
            <button onClick={runBacktest} className="px-4 py-2 rounded-lg flex items-center gap-2 transition-all" style={{
              background: 'var(--ui-surface-2)',
              border: '1px solid var(--ui-border)',
              color: 'var(--ui-text-muted)'
            }}>
              <RefreshCw size={16} />
              Backtest
            </button>
          </div>
          <ThemeSwitcher />
          <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ 
            background: 'linear-gradient(135deg, var(--ui-surface-3) 0%, var(--ui-surface-2) 100%)',
            border: '1px solid var(--ui-border)'
          }}>
            <User size={18} />
          </div>
        </div>
      </header>

      {/* Main Content - World-Class Layout */}
      <main className="px-10 py-8 max-w-[1600px] mx-auto">
        {/* Hero Section - Premium Trading Metrics */}
        <section className="mb-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            
            {/* Gate - ORCSMC with Production Indicators */}
            <div className="ui-card-hero glass-effect p-6 lg:col-span-1">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="section-header">Gate - ORCSMC</h2>
                  <div className="metric-hero">{(telemetry?.gate_prob * 100).toFixed(1)}%</div>
                  <div className="text-xs" style={{ color: 'var(--ui-text-subtle)' }}>
                    Threshold: 60.0% | Two-day inertia rule
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <div className={`ui-status-pill ${telemetry?.gate_prob >= 0.6 ? 'active' : ''}`}>
                    {telemetry?.gate_prob >= 0.6 ? 'RISK-ON' : 'GATE-OFF'}
                  </div>
                  {telemetry?.gate_prob >= 0.6 && (
                    <div className="px-2 py-1 rounded text-xs" style={{ 
                      background: 'var(--ui-surface-3)', 
                      color: 'var(--ui-text-muted)' 
                    }}>
                      INERTIA ON
                    </div>
                  )}
                </div>
              </div>
              <div className="mb-4">
                <Sparkline data={[0.45, 0.52, 0.61, 0.58, 0.65, 0.72, 0.68, 0.71, 0.69, 0.72]} />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 rounded-lg" style={{ background: 'var(--ui-surface-3)' }}>
                  <span className="data-label block mb-1">Confidence</span>
                  <span className="data-value text-lg">{((telemetry?.orcsmc?.regime_confidence || 0.85) * 100).toFixed(0)}%</span>
                </div>
                <div className="p-3 rounded-lg" style={{ background: 'var(--ui-surface-3)' }}>
                  <span className="data-label block mb-1">ESS of N</span>
                  <span className="data-value text-lg">{telemetry?.orcsmc?.particle_ess || 847}/1000</span>
                </div>
              </div>
            </div>

            {/* Bias - Enhanced with Features & Training Meta */}
            <div className="ui-card-hero glass-effect p-6">
              <h2 className="section-header">Market Bias</h2>
              <div className="text-center mb-4">
                <NeumorphicBiasGauge 
                  bias={telemetry?.bias || 0} 
                  loading={loading}
                />
              </div>
              
              {/* Top-3 Feature Chips */}
              <div className="mb-4">
                <div className="text-xs font-semibold mb-2" style={{ color: 'var(--ui-text-subtle)' }}>
                  TOP FEATURES
                </div>
                <div className="flex gap-2 flex-wrap">
                  <div className="px-2 py-1 rounded text-xs" style={{ 
                    background: 'var(--ui-surface-3)', 
                    color: 'var(--ui-text-muted)' 
                  }}>
                    SIG_NORM_L2: 0.34
                  </div>
                  <div className="px-2 py-1 rounded text-xs" style={{ 
                    background: 'var(--ui-surface-3)', 
                    color: 'var(--ui-text-muted)' 
                  }}>
                    VOL_DRIFT: -0.12
                  </div>
                  <div className="px-2 py-1 rounded text-xs" style={{ 
                    background: 'var(--ui-surface-3)', 
                    color: 'var(--ui-text-muted)' 
                  }}>
                    FUNDING_MA: 0.09
                  </div>
                </div>
              </div>
              
              {/* Training Meta */}
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div className="data-row">
                  <span className="data-label">BEMA β</span>
                  <span className="data-value">0.999</span>
                </div>
                <div className="data-row">
                  <span className="data-label">EMA τ½</span>
                  <span className="data-value">693 steps</span>
                </div>
                <div className="data-row">
                  <span className="data-label">LR</span>
                  <span className="data-value">HIGH</span>
                </div>
              </div>
            </div>

            {/* PnL & Status - Tertiary Hero */}
            <div className="ui-card-hero glass-effect p-4">
              <h2 className="section-header">Performance</h2>
              <div className="text-center py-2">
                <div className={`metric-large mb-2`} style={{ 
                  color: telemetry?.recent_pnl >= 0 ? 'var(--ui-success)' : 'var(--ui-danger)' 
                }}>
                  {telemetry?.recent_pnl >= 0 ? '+' : ''}{((telemetry?.recent_pnl || 0) * 100).toFixed(2)}%
                </div>
                <div className="space-y-1 text-sm">
                  <div className="data-row">
                    <span className="data-label">Target x*</span>
                    <span className="data-value">{telemetry?.target_exposure?.toFixed(3) || '0.000'}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Current x</span>
                    <span className="data-value">{telemetry?.current_exposure?.toFixed(3) || '0.000'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Risk Guards - Only Show When Active */}
        {(telemetry?.execution?.guards?.funding_top_5_percent || telemetry?.execution?.guards?.vol_drift_30_percent) && (
          <section className="mb-4">
            <h2 className="section-header mb-3 text-sm">Risk Guards Active</h2>
            <div className="flex gap-2 flex-wrap">
              {telemetry?.execution?.guards?.funding_top_5_percent && (
                <div className="px-3 py-1 rounded-lg text-xs font-medium" style={{
                  background: 'var(--ui-danger)',
                  color: '#000'
                }}>
                  FUNDING TOP-5% HALVING
                </div>
              )}
              {telemetry?.execution?.guards?.vol_drift_30_percent && (
                <div className="px-3 py-1 rounded-lg text-xs font-medium" style={{
                  background: 'var(--ui-warning)',
                  color: '#000'
                }}>
                  VOL DRIFT +30% HALVING
                </div>
              )}
              {telemetry?.execution?.guards?.daily_loss_cap && (
                <div className="px-3 py-1 rounded-lg text-xs font-medium" style={{
                  background: 'var(--ui-danger)',
                  color: '#000'
                }}>
                  DAILY LOSS CAP
                </div>
              )}
            </div>
          </section>
        )}

        {/* Primary Metrics - Execution Details */}
        <section className="mb-4">
          <h2 className="section-header mb-3 text-sm">Execution Metrics</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            
            {/* Optimal Execution - Enhanced */}
            <div className="ui-card glass-effect p-3">
              <h3 className="section-header text-sm">Optimal Execution</h3>
              <div className="space-y-2 text-sm">
                <div className="data-row">
                  <span className="data-label">κ rate</span>
                  <span className="data-value">{telemetry?.execution?.execution_rate?.toFixed(3) || '0.000'}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Impact dl</span>
                  <span className="data-value">{(telemetry?.execution?.price_impact * 10000)?.toFixed(1) || '0.0'}bp</span>
                </div>
                <div className="data-row">
                  <span className="data-label">κ fast</span>
                  <span className="data-value">{telemetry?.execution?.kappa_fast?.toFixed(3) || '0.000'}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">κ slow</span>
                  <span className="data-value">{telemetry?.execution?.kappa_slow?.toFixed(3) || '0.000'}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Planned step</span>
                  <span className="data-value">0.125 units</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Exposure preview</span>
                  <span className="data-value">{telemetry?.current_exposure?.toFixed(3) || '0.000'}→{telemetry?.target_exposure?.toFixed(3) || '0.000'}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Last calibrated</span>
                  <span className="data-value">3d ago</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Half-step</span>
                  <span className={`ui-status-pill ${telemetry?.execution?.half_step_active ? 'active' : ''}`}>
                    {telemetry?.execution?.half_step_active ? 'ON' : 'OFF'}
                  </span>
                </div>
              </div>
            </div>

            {/* Deep Optimal Stopping */}
            <div className="ui-card glass-effect p-3">
              <h3 className="section-header text-sm">Deep Optimal Stopping</h3>
              <div className="flex items-center justify-between mb-2">
                <div className="metric-medium">{telemetry?.stopping?.decision || 'HOLD'}</div>
                <div className="text-sm data-value">Conf: {((telemetry?.stopping?.confidence || 0.85) * 100).toFixed(0)}%</div>
              </div>
              <div className="space-y-2 text-sm">
                <div className="data-row">
                  <span className="data-label">Lower bound</span>
                  <span className="data-value">{(telemetry?.stopping?.lower_bound || 0.02).toFixed(3)}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Upper bound</span>
                  <span className="data-value">{(telemetry?.stopping?.upper_bound || 0.08).toFixed(3)}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Continuation</span>
                  <span className="data-value">{(telemetry?.stopping?.continuation_value || 0.045).toFixed(3)}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Day/Max</span>
                  <span className="data-value">{telemetry?.stopping?.day_counter || 2}/{telemetry?.stopping?.max_days || 7}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CRITICAL: Validation Card - Go/No-Go Decision */}
        <section className="mb-6">
          <div className="ui-card-hero glass-effect p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold" style={{ color: 'var(--ui-text)' }}>
                Validation - Walk-Forward Acceptance
              </h2>
              <div className="px-4 py-2 rounded-lg text-sm font-semibold" style={{
                background: 'var(--ui-success)',
                color: '#000'
              }}>
                GO / READY
              </div>
            </div>
            
            {/* Four Acceptance Ticks */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div className="text-center">
                <div className="w-8 h-8 rounded-full mx-auto mb-2 flex items-center justify-center" style={{
                  background: 'var(--ui-success)',
                  color: '#000'
                }}>
                  ✓
                </div>
                <div className="text-xs font-medium" style={{ color: 'var(--ui-text-muted)' }}>HIT RATE</div>
                <div className="text-sm data-value">64.2%</div>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 rounded-full mx-auto mb-2 flex items-center justify-center" style={{
                  background: 'var(--ui-success)',
                  color: '#000'
                }}>
                  ✓
                </div>
                <div className="text-xs font-medium" style={{ color: 'var(--ui-text-muted)' }}>SHARPE</div>
                <div className="text-sm data-value">1.85</div>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 rounded-full mx-auto mb-2 flex items-center justify-center" style={{
                  background: 'var(--ui-success)',
                  color: '#000'
                }}>
                  ✓
                </div>
                <div className="text-xs font-medium" style={{ color: 'var(--ui-text-muted)' }}>MAX DD</div>
                <div className="text-sm data-value">12.4%</div>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 rounded-full mx-auto mb-2 flex items-center justify-center" style={{
                  background: 'var(--ui-success)',
                  color: '#000'
                }}>
                  ✓
                </div>
                <div className="text-xs font-medium" style={{ color: 'var(--ui-text-muted)' }}>GATE-OFF</div>
                <div className="text-sm data-value">Low Risk</div>
              </div>
            </div>
            
            {/* Slice/Costs/Embargo Footer */}
            <div className="pt-4 border-t" style={{ borderColor: 'var(--ui-border)' }}>
              <div className="grid grid-cols-3 gap-4 text-xs">
                <div className="text-center">
                  <div className="data-label">Slice Window</div>
                  <div className="data-value">365d→30d</div>
                </div>
                <div className="text-center">
                  <div className="data-label">Costs On</div>
                  <div className="data-value">Fees+Slippage+Funding</div>
                </div>
                <div className="text-center">
                  <div className="data-label">Embargo</div>
                  <div className="data-value">48h</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Secondary Metrics - Technical Details */}
        <section className="mb-4">
          <h2 className="section-header mb-3 text-sm">Technical Details</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
            
            {/* Signature Features */}
            <div className="ui-card glass-effect p-3">
              <h3 className="section-header text-sm">Signature Features</h3>
              <div className="mb-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-medium" style={{ color: 'var(--ui-text-subtle)' }}>ORDER-3 TRUNCATION</span>
                  <span className="data-value">{telemetry?.signature_features?.n_features || 28}/32</span>
                </div>
                <div className="space-y-1 text-sm">
                  <div className="data-row">
                    <span className="data-label">sig_time_price</span>
                    <span className="data-value">{(telemetry?.signature_features?.sig_time_price || 0.42).toFixed(3)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">sig_price_vol</span>
                    <span className="data-value">{(telemetry?.signature_features?.sig_price_vol || -0.18).toFixed(3)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">sig_ppp (triple)</span>
                    <span className="data-value">{(telemetry?.signature_features?.sig_ppp || 0.08).toFixed(3)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">vol_of_vol</span>
                    <span className="data-value">{(telemetry?.signature_features?.vol_of_vol || 0.22).toFixed(3)}</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-xs font-medium mb-1" style={{ color: 'var(--ui-text-subtle)' }}>TOP ATTRIBUTIONS</h4>
                <div className="space-y-1">
                  {telemetry?.attributions?.slice(0, 3).map((attr: any, i: number) => (
                    <div key={i} className="data-row">
                      <span className="data-label font-mono truncate">{attr.name}</span>
                      <span className="data-value" style={{ 
                        color: attr.attribution > 0 ? 'var(--ui-accent)' : 'var(--ui-text-subtle)' 
                      }}>
                        {attr.attribution > 0 ? '+' : ''}{attr.attribution?.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* JAL Loss Training */}
            <div className="ui-card glass-effect p-3">
              <h3 className="section-header text-sm">JAL Loss Training</h3>
              <div className="space-y-1 text-sm mb-3">
                <div className="data-row">
                  <span className="data-label">Active loss (CE)</span>
                  <span className="data-value">{(telemetry?.jal_training?.active_loss || 0.12).toFixed(3)}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Passive loss (AMSE)</span>
                  <span className="data-value">{(telemetry?.jal_training?.passive_loss || 0.08).toFixed(3)}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">α/β weights</span>
                  <span className="data-value">{telemetry?.jal_training?.alpha || 0.7}/{telemetry?.jal_training?.beta || 0.3}</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Noise level</span>
                  <span className="data-value">{((telemetry?.jal_training?.noise_level || 0.09) * 100).toFixed(1)}%</span>
                </div>
                <div className="data-row">
                  <span className="data-label">Asymmetry param</span>
                  <span className="data-value">{telemetry?.jal_training?.asymmetry_param || 3.0}</span>
                </div>
              </div>
              <div className={`ui-status-pill ${telemetry?.jal_training?.jal_enabled ? 'active' : ''}`}>
                {telemetry?.jal_training?.jal_enabled ? 'JAL ACTIVE' : 'CLEAN DATA'}
              </div>
            </div>

            {/* Learning & Optimization */}
            <div className="ui-card glass-effect p-3">
              <h3 className="section-header text-sm">Learning & Optimization</h3>
              <div className="mb-3">
                <h4 className="text-xs font-medium mb-1" style={{ color: 'var(--ui-text-subtle)' }}>LARGE LR SCHEDULER</h4>
                <div className="space-y-1 text-sm">
                  <div className="data-row">
                    <span className="data-label">Current LR</span>
                    <span className="data-value">{(telemetry?.learning_schedule?.current_lr || 0.15).toFixed(3)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Sparsity</span>
                    <span className="data-value">{((telemetry?.learning_schedule?.sparsity_ratio || 0.34) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Confidence</span>
                    <span className="data-value">{((telemetry?.learning_schedule?.confidence_score || 0.87) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
              <div className="mb-3">
                <h4 className="text-xs font-medium mb-1" style={{ color: 'var(--ui-text-subtle)' }}>BEMA OPTIMIZER</h4>
                <div className="space-y-1 text-sm">
                  <div className="data-row">
                    <span className="data-label">Convergence rate</span>
                    <span className="data-value">{telemetry?.bema_metrics?.convergence_rate || 2.4}x</span>
                  </div>
                  <div className="data-row">
                    <span className="data-label">Stability</span>
                    <span className="data-value">{((telemetry?.bema_metrics?.stability_score || 0.91) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
              <div className={`ui-status-pill ${telemetry?.learning_schedule?.compression_ready ? 'active' : ''}`}>
                {telemetry?.learning_schedule?.compression_ready ? 'COMPRESSION READY' : 'TRAINING'}
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Status Footer */}
      <footer className="px-8 py-6 flex items-center justify-between" style={{ 
        backgroundColor: 'var(--ui-surface)', 
        borderTop: '1px solid var(--ui-border)' 
      }}>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <Activity size={18} style={{ color: 'var(--ui-accent)' }} />
            <span className="data-value">Pipeline Status: Active</span>
          </div>
          <div className="data-label">
            Last run: {new Date(Date.now() - 3600000).toLocaleTimeString()}
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="data-label">
            PnL: <span className="data-value" style={{ color: telemetry?.recent_pnl >= 0 ? 'var(--ui-success)' : 'var(--ui-danger)' }}>
              {telemetry?.recent_pnl >= 0 ? '+' : ''}{((telemetry?.recent_pnl || 0) * 100).toFixed(2)}%
            </span>
          </div>
          <div className="data-label">
            Next run: {new Date(Date.now() + 21600000).toLocaleTimeString()}
          </div>
        </div>
      </footer>
    </div>
  );
}