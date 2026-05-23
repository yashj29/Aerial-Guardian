import { useState, useEffect, useRef, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import './index.css'

const API = ''   // proxied via vite → localhost:8000

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────────────────────────────────────
function Sidebar({ sequences, selected, onSelect }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>Aerial Guardian</h1>
        <p>VisDrone MOT · Person Tracking</p>
      </div>
      <div className="seq-list">
        {sequences.map(seq => (
          <div
            key={seq.name}
            className={`seq-item ${selected?.name === seq.name ? 'active' : ''}`}
            onClick={() => onSelect(seq)}
          >
            <div className={`seq-dot ${seq.is_running ? 'running' : seq.has_video ? 'ready' : ''}`} />
            <span className="seq-name">{seq.name}</span>
            {seq.avg_fps && <span className="seq-fps">{seq.avg_fps} fps</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Video Tab
// ─────────────────────────────────────────────────────────────────────────────
function VideoTab({ seq }) {
  if (!seq.has_video) {
    return (
      <div className="empty">
        <p>No processed video yet. Use "Run Pipeline" to generate it.</p>
      </div>
    )
  }
  // Key on seq.name forces the <video> to reload when sequence changes
  return (
    <div className="video-wrap">
      <video key={seq.name} controls autoPlay muted loop>
        <source src={`${API}/api/videos/${seq.name}`} type="video/mp4" />
      </video>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Live Inference Tab
// ─────────────────────────────────────────────────────────────────────────────
function LiveTab({ seq }) {
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ frame: 0, total: 0 })
  const [stats, setStats] = useState({ fps: 0, n_dets: 0, n_tracks: 0 })

  const stop = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send('stop')
      wsRef.current.close()
      wsRef.current = null
    }
    setRunning(false)
  }, [])

  const start = useCallback(() => {
    if (running) { stop(); return }
    setRunning(true)
    setProgress({ frame: 0, total: 0 })

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/live/${seq.name}`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.done) { setRunning(false); return }
      if (msg.error) { console.error(msg.error); setRunning(false); return }

      setProgress({ frame: msg.frame_idx + 1, total: msg.total })
      setStats({ fps: msg.fps, n_dets: msg.n_dets, n_tracks: msg.n_tracks })

      const img = new Image()
      img.onload = () => {
        const canvas = canvasRef.current
        if (!canvas) return
        canvas.width = img.width
        canvas.height = img.height
        canvas.getContext('2d').drawImage(img, 0, 0)
      }
      img.src = 'data:image/jpeg;base64,' + msg.image
    }

    ws.onclose = () => setRunning(false)
    ws.onerror = () => setRunning(false)
  }, [seq.name, running, stop])

  useEffect(() => () => stop(), [seq.name])

  const pct = progress.total > 0 ? (progress.frame / progress.total) * 100 : 0

  return (
    <div className="live-wrap">
      <div className="live-canvas-wrap">
        <canvas ref={canvasRef} style={{ minHeight: 200, background: '#000' }} />
        {running && (
          <div className="live-overlay">
            <div className="live-chip">FPS <span>{stats.fps}</span></div>
            <div className="live-chip">Tracks <span>{stats.n_tracks}</span></div>
            <div className="live-chip">Dets <span>{stats.n_dets}</span></div>
          </div>
        )}
      </div>

      <div className="progress-bar-wrap">
        <div className="progress-bar" style={{ width: `${pct}%` }} />
      </div>

      <div className="live-controls">
        <button className={`btn ${running ? 'btn-danger' : 'btn-primary'}`} onClick={start}>
          {running ? 'Stop' : 'Start Live Inference'}
        </button>
        {progress.total > 0 && (
          <span style={{ fontSize: 11, color: '#64748b' }}>
            Frame {progress.frame} / {progress.total}
          </span>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Compare Tab
// ─────────────────────────────────────────────────────────────────────────────
function CompareTab({ seq }) {
  const [frameIdx, setFrameIdx] = useState(0)
  const [totalFrames, setTotalFrames] = useState(100)

  useEffect(() => {
    fetch(`${API}/api/metrics/${seq.name}`)
      .then(r => r.json())
      .then(m => { if (m.frames) setTotalFrames(m.frames) })
      .catch(() => {})
  }, [seq.name])

  if (!seq.has_video) {
    return (
      <div className="empty">
        <p>Run the pipeline first to enable frame comparison.</p>
      </div>
    )
  }

  const orig = `${API}/api/frames/original/${seq.name}/${frameIdx}`
  const tracked = `${API}/api/frames/tracked/${seq.name}/${frameIdx}`

  return (
    <div className="compare-wrap">
      <div className="scrubber-row">
        <span>Frame {frameIdx + 1} / {totalFrames}</span>
        <input
          className="scrubber"
          type="range" min={0} max={totalFrames - 1}
          value={frameIdx}
          onChange={e => setFrameIdx(Number(e.target.value))}
        />
      </div>
      <div className="compare-frames">
        <div className="frame-box">
          <div className="frame-label">Original</div>
          <img src={orig} alt="original frame" />
        </div>
        <div className="frame-box">
          <div className="frame-label">Tracked (YOLOv8n + BoT-SORT)</div>
          <img src={tracked} alt="tracked frame" />
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics Tab
// ─────────────────────────────────────────────────────────────────────────────
function MetricsTab({ seq }) {
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    setMetrics(null)
    fetch(`${API}/api/metrics/${seq.name}`)
      .then(r => r.json())
      .then(setMetrics)
      .catch(() => {})
  }, [seq.name])

  if (!metrics) {
    return (
      <div className="empty">
        <p>{seq.has_video ? 'Loading metrics…' : 'Run the pipeline to generate metrics.'}</p>
      </div>
    )
  }

  // Downsample per_frame to max 300 points for the chart
  const pf = metrics.per_frame || []
  const step = Math.max(1, Math.floor(pf.length / 300))
  const chartData = pf
    .filter((_, i) => i % step === 0)
    .map((f, i) => ({ frame: i * step, fps: f.fps, tracks: f.n_tracks, dets: f.n_dets }))

  const avgTracks = pf.length ? (pf.reduce((s, f) => s + f.n_tracks, 0) / pf.length).toFixed(1) : 0
  const maxTracks = pf.length ? Math.max(...pf.map(f => f.n_tracks)) : 0

  return (
    <div>
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="label">Avg FPS</div>
          <div className="value">{metrics.avg_fps}<span className="unit">fps</span></div>
        </div>
        <div className="metric-card">
          <div className="label">Total Frames</div>
          <div className="value">{metrics.frames}</div>
        </div>
        <div className="metric-card">
          <div className="label">Avg Tracks / Frame</div>
          <div className="value">{avgTracks}</div>
        </div>
        <div className="metric-card">
          <div className="label">Peak Tracks</div>
          <div className="value">{maxTracks}</div>
        </div>
        <div className="metric-card">
          <div className="label">Resolution</div>
          <div className="value" style={{ fontSize: 16 }}>{metrics.width}×{metrics.height}</div>
        </div>
        <div className="metric-card">
          <div className="label">Sequence</div>
          <div className="value" style={{ fontSize: 12, paddingTop: 6, color: '#94a3b8' }}>{metrics.sequence}</div>
        </div>
      </div>

      <div className="chart-card">
        <h3>FPS over Time</h3>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2d48" />
            <XAxis dataKey="frame" tick={{ fontSize: 10, fill: '#475569' }} />
            <YAxis tick={{ fontSize: 10, fill: '#475569' }} />
            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1f2d48', fontSize: 11 }} />
            <Line type="monotone" dataKey="fps" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-card">
        <h3>Active Tracks &amp; Detections over Time</h3>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2d48" />
            <XAxis dataKey="frame" tick={{ fontSize: 10, fill: '#475569' }} />
            <YAxis tick={{ fontSize: 10, fill: '#475569' }} />
            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1f2d48', fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="tracks" stroke="#22c55e" dot={false} strokeWidth={1.5} name="Tracks" />
            <Line type="monotone" dataKey="dets" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="Detections" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// App root
// ─────────────────────────────────────────────────────────────────────────────
const TABS = ['Video', 'Live', 'Compare', 'Metrics']

export default function App() {
  const [sequences, setSequences] = useState([])
  const [selected, setSelected] = useState(null)
  const [tab, setTab] = useState('Video')

  const fetchSequences = useCallback(() => {
    fetch(`${API}/api/sequences`)
      .then(r => r.json())
      .then(data => {
        setSequences(data)
        setSelected(prev => prev ? data.find(s => s.name === prev.name) || prev : data[0] || null)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetchSequences()
    const id = setInterval(fetchSequences, 3000)
    return () => clearInterval(id)
  }, [fetchSequences])

  const runPipeline = () => {
    if (!selected) return
    fetch(`${API}/api/run/${selected.name}`, { method: 'POST' })
      .then(() => fetchSequences())
  }

  const seqStatus = selected?.is_running ? 'amber' : selected?.has_video ? 'green' : ''

  return (
    <div className="app">
      <Sidebar sequences={sequences} selected={selected} onSelect={seq => { setSelected(seq); setTab('Video') }} />

      <div className="main">
        {selected ? (
          <>
            <div className="topbar">
              <h2>{selected.name}</h2>
              {seqStatus && (
                <span className={`badge ${seqStatus}`}>
                  {selected.is_running ? 'Processing…' : 'Ready'}
                </span>
              )}
              <button
                className="btn btn-primary"
                onClick={runPipeline}
                disabled={selected.is_running}
              >
                {selected.is_running ? 'Running…' : 'Run Pipeline'}
              </button>
            </div>

            <div className="tabs">
              {TABS.map(t => (
                <div key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</div>
              ))}
            </div>

            <div className="content">
              {tab === 'Video'   && <VideoTab   seq={selected} />}
              {tab === 'Live'    && <LiveTab    seq={selected} key={selected.name} />}
              {tab === 'Compare' && <CompareTab seq={selected} />}
              {tab === 'Metrics' && <MetricsTab seq={selected} />}
            </div>
          </>
        ) : (
          <div className="empty" style={{ height: '100vh' }}>
            <p>Select a sequence from the sidebar</p>
          </div>
        )}
      </div>
    </div>
  )
}
