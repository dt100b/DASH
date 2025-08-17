import React, { useState, useEffect } from 'react'
import { 
  TrendingUp, TrendingDown, Activity, AlertCircle,
  Play, FileText, Upload, Download, RefreshCw
} from 'lucide-react'
import { LineChart, Line, AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts'
import HeaderMeta from './components/HeaderMeta'
import GateCard from './components/GateCard'
import BiasCard from './components/BiasCard'
import SizingCard from './components/SizingCard'
import StopsCard from './components/StopsCard'
import ValidationCard from './components/ValidationCard'

// Radial gauge component
const RadialGauge = ({ value, label, threshold = 0.6 }) => {
  const radius = 45
  const strokeWidth = 3
  const normalizedValue = Math.min(Math.max(value, 0), 1)
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (normalizedValue * circumference)
  
  return (
    <div className="relative w-32 h-32">
      <svg className="transform -rotate-90 w-32 h-32">
        <circle
          cx="64"
          cy="64"
          r={radius}
          stroke="var(--ink-1)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        <circle
          cx="64"
          cy="64"
          r={radius}
          stroke={value >= threshold ? "var(--acc-1)" : "var(--ink-2)"}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="numeric-display text-3xl text-txt-1">{(value * 100).toFixed(0)}%</span>
        <span className="text-xs text-txt-3 mt-1">{label}</span>
      </div>
    </div>
  )
}

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
      <div className="absolute bottom-0 left-0 text-xs text-txt-3">SHORT</div>
      <div className="absolute bottom-0 right-0 text-xs text-txt-3">LONG</div>
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 text-xs text-txt-3">FLAT</div>
    </div>
  )
}

// Sparkline component
const Sparkline = ({ data, color = "var(--acc-1)" }) => {
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * 100
    const y = ((max - val) / range) * 40 + 5
    return `${x},${y}`
  }).join(' ')
  
  return (
    <svg className="w-full h-12" viewBox="0 0 100 50">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1"
        opacity="0.8"
      />
      <line x1="0" y1="25" x2="100" y2="25" stroke="var(--ink-1)" strokeWidth="0.5" strokeDasharray="2,2" />
    </svg>
  )
}

// Main App component
function App() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC')
  const [telemetryData, setTelemetryData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [time, setTime] = useState(new Date())
  
  // FORCE BROWSER UPDATE
  useEffect(() => {
    console.log('🔥 NEW THIN HEADER VERSION LOADED 🔥', new Date().toISOString())
  }, [])

  // Update clock
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Fetch telemetry data
  useEffect(() => {
    fetchTelemetry()
    const interval = setInterval(fetchTelemetry, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [selectedSymbol])

  const fetchTelemetry = async () => {
    try {
      const response = await fetch(`/api/telemetry?symbol=${selectedSymbol}`)
      const data = await response.json()
      setTelemetryData(data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch telemetry:', error)
      setTelemetryData(null)
      setLoading(false)
    }
  }

  const runNow = async () => {
    try {
      const response = await fetch('/api/run-now', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: selectedSymbol })
      })
      const data = await response.json()
      console.log('Run completed:', data)
      fetchTelemetry() // Refresh data
    } catch (error) {
      console.error('Run failed:', error)
    }
  }

  const runBacktest = async () => {
    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: selectedSymbol, days: 30 })
      })
      const data = await response.json()
      console.log('Backtest completed:', data)
    } catch (error) {
      console.error('Backtest failed:', error)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-bg-0 flex items-center justify-center">
        <div className="text-txt-3 animate-pulse">Loading dashboard...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg-0 p-4" key={Date.now()}>
      {/* ULTRA THIN HEADER */}
      <header className="flex items-center justify-between mb-1 px-2 py-1" style={{borderBottom: '3px solid red', height: '35px', background: 'rgba(255,0,0,0.1)'}}>
        <div className="flex items-center gap-4">
          {/* Symbol selector */}
          <div className="flex gap-1">
            {['BTC', 'ETH'].map(symbol => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                className={`px-3 py-1 rounded text-sm font-light transition-all ${
                  selectedSymbol === symbol 
                    ? 'bg-bg-2 text-txt-1 border border-ink-2' 
                    : 'bg-bg-1 text-txt-3 border border-ink-1 hover:bg-bg-2'
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>
          
          {/* UTC Clock - FORCED UPDATE */}
          <div className="text-white text-xs font-light bg-red-600 px-3 py-2 rounded-full border-2 border-yellow-400">
            ⚡ THIN HEADER ⚡ {time.toUTCString().slice(17, 25)} UTC
          </div>

          {/* Header meta */}
          <HeaderMeta meta={telemetryData?.meta} />
        </div>
        
        {/* Gate status */}
        <div className="flex items-center gap-2">
          <span className="text-txt-3 text-xs">GATE</span>
          <div className="relative w-16 h-16">
            <svg className="transform -rotate-90 w-16 h-16">
              <circle
                cx="32"
                cy="32"
                r="24"
                stroke="var(--ink-1)"
                strokeWidth="2"
                fill="none"
              />
              <circle
                cx="32"
                cy="32"
                r="24"
                stroke={(telemetryData?.gate?.p_risk_on || 0) >= (telemetryData?.gate?.threshold || 0.6) ? "var(--acc-1)" : "var(--ink-2)"}
                strokeWidth="2"
                fill="none"
                strokeDasharray={150.8}
                strokeDashoffset={150.8 - ((telemetryData?.gate?.p_risk_on || 0) * 150.8)}
                className="transition-all duration-500"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="numeric-display text-lg text-txt-1">{((telemetryData?.gate?.p_risk_on || 0) * 100).toFixed(0)}%</span>
              <span className="text-xs text-txt-3 mt-0">RISK-ON</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main grid */}
      <div className="grid grid-cols-2 gap-4 max-w-7xl mx-auto">
        {/* Left column - State */}
        <div className="space-y-4">
          <GateCard gate={telemetryData?.gate} />
          <BiasCard bias={telemetryData?.bias} />
          <SizingCard sizing={telemetryData?.sizing} />
        </div>

        {/* Right column - Diagnostics */}
        <div className="space-y-4">
          <StopsCard stopping={telemetryData?.stopping} />
          <ValidationCard validation={telemetryData?.validation} />
        </div>
      </div>

      {/* Footer action bar */}
      <footer className="fixed bottom-0 left-0 right-0 bg-bg-1 border-t border-ink-1 p-4">
        <div className="max-w-7xl mx-auto flex gap-3 justify-center">
          <button
            onClick={runNow}
            className="px-4 py-2 bg-bg-2 text-txt-1 border border-ink-2 rounded-lg flex items-center gap-2 hover:bg-bg-3 transition-all"
          >
            <Play size={14} />
            Run Now
          </button>
          <button
            onClick={runBacktest}
            className="px-4 py-2 bg-bg-2 text-txt-1 border border-ink-2 rounded-lg flex items-center gap-2 hover:bg-bg-3 transition-all"
          >
            <RefreshCw size={14} />
            Backtest 30d
          </button>
          <button className="px-4 py-2 bg-bg-2 text-txt-1 border border-ink-2 rounded-lg flex items-center gap-2 hover:bg-bg-3 transition-all">
            <Upload size={14} />
            Import Weights
          </button>
          <button className="px-4 py-2 bg-bg-2 text-txt-1 border border-ink-2 rounded-lg flex items-center gap-2 hover:bg-bg-3 transition-all">
            <Download size={14} />
            Export Logs
          </button>
        </div>
      </footer>
    </div>
  )
}

export default App