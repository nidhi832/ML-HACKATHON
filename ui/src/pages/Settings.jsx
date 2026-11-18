import React, { useState, useEffect } from 'react';
import { 
  Settings as SettingsIcon, 
  Cpu, 
  Map, 
  Palette, 
  Radio, 
  Check, 
  Server,
  Layers,
  Wifi,
  AlertCircle
} from 'lucide-react';
import { getApiBaseUrl, setApiBaseUrl, fetchHealth } from '../services/api';

export default function Settings() {
  const [defaultModel, setDefaultModel] = useState('lightgbm');
  const [candidateCount, setCandidateCount] = useState(10);
  const [autoCenter, setAutoCenter] = useState(true);
  const [matchedColor, setMatchedColor] = useState('#3B82F6');
  const [refColor, setRefColor] = useState('#10B981');
  const [gpsColor, setGpsColor] = useState('#EF4444');
  const [saved, setSaved] = useState(false);

  // Two-laptop deployment connection config
  const [apiUrl, setApiUrl] = useState('');
  const [connStatus, setConnStatus] = useState('idle'); // 'idle' | 'checking' | 'connected' | 'error'
  const [connMsg, setConnMsg] = useState('');

  useEffect(() => {
    setApiUrl(getApiBaseUrl());
    testConnection(getApiBaseUrl());
  }, []);

  const testConnection = async (urlToTest) => {
    setConnStatus('checking');
    setConnMsg('Connecting to API server...');
    const prev = getApiBaseUrl();
    try {
      setApiBaseUrl(urlToTest);
      const res = await fetchHealth();
      setConnStatus('connected');
      setConnMsg(`Connected: ${res.system} (v${res.version}) - Model: ${res.model_name}`);
    } catch (e) {
      setConnStatus('error');
      setConnMsg(`Connection Failed: ${e.message}`);
      setApiBaseUrl(prev);
    }
  };

  const handleSave = () => {
    setApiBaseUrl(apiUrl);
    testConnection(apiUrl);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="p-6 space-y-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Platform Settings</h1>
        <p className="text-sm text-slate-400 mt-1">
          Configure default pipeline models, candidate generation thresholds, and visualization preferences.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Machine Learning Pipeline Defaults */}
        <div className="bg-panel border border-border rounded-card p-5 space-y-5 shadow-xl">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <Cpu className="w-4 h-4 text-primary" />
            <h3 className="font-semibold text-white text-sm">Pipeline Defaults</h3>
          </div>

          <div className="space-y-4 text-xs">
            <div className="space-y-1.5">
              <label className="text-slate-300 font-semibold block">Default Ranking Model</label>
              <select
                value={defaultModel}
                onChange={(e) => setDefaultModel(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-primary transition"
              >
                <option value="lightgbm">LightGBM (Gradient Boosting) [Champion]</option>
                <option value="random_forest">Random Forest (Bagging Ensemble)</option>
                <option value="pytorch_mlp">PyTorch Deep MLP Neural Network</option>
                <option value="logistic_regression">Logistic Regression (L2 Scaled)</option>
                <option value="subsampled_knn">K-Nearest Neighbours</option>
                <option value="gaussian_nb">Gaussian Naive Bayes</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <label className="text-slate-300 font-semibold block">Default Candidate Pool (K)</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={5}
                  max={20}
                  step={5}
                  value={candidateCount}
                  onChange={(e) => setCandidateCount(parseInt(e.target.value, 10))}
                  className="flex-1 h-1.5 bg-background rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <span className="font-mono font-bold text-primary w-12 text-right">K = {candidateCount}</span>
              </div>
              <p className="text-[11px] text-slate-500 font-mono">
                K=10 achieves 98.83% spatial recall on benchmark trips.
              </p>
            </div>
          </div>
        </div>

        {/* Map Visualization & Styles */}
        <div className="bg-panel border border-border rounded-card p-5 space-y-5 shadow-xl">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <Palette className="w-4 h-4 text-primary" />
            <h3 className="font-semibold text-white text-sm">Map & Layer Colors</h3>
          </div>

          <div className="space-y-4 text-xs">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-slate-200 font-semibold block">Matched Route Color</span>
                <span className="text-[11px] text-slate-500">PathSnap model output polyline</span>
              </div>
              <input
                type="color"
                value={matchedColor}
                onChange={(e) => setMatchedColor(e.target.value)}
                className="w-8 h-8 rounded border border-border cursor-pointer bg-transparent"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <span className="text-slate-200 font-semibold block">Reference Route Color</span>
                <span className="text-[11px] text-slate-500">Ground truth verified path</span>
              </div>
              <input
                type="color"
                value={refColor}
                onChange={(e) => setRefColor(e.target.value)}
                className="w-8 h-8 rounded border border-border cursor-pointer bg-transparent"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <span className="text-slate-200 font-semibold block">Raw GPS Point Color</span>
                <span className="text-[11px] text-slate-500">Raw noisy observation marker</span>
              </div>
              <input
                type="color"
                value={gpsColor}
                onChange={(e) => setGpsColor(e.target.value)}
                className="w-8 h-8 rounded border border-border cursor-pointer bg-transparent"
              />
            </div>
          </div>
        </div>

        {/* System & Two-Laptop Distributed Connection */}
        <div className="bg-panel border border-border rounded-card p-5 space-y-4 shadow-xl md:col-span-2">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-primary" />
              <h3 className="font-semibold text-white text-sm">Two-Laptop Distributed Connection</h3>
            </div>
            <div className="flex items-center gap-2">
              {connStatus === 'checking' && (
                <span className="text-xs text-yellow-400 flex items-center gap-1 font-mono">
                  <Wifi className="w-3.5 h-3.5 animate-spin" /> Probing /health...
                </span>
              )}
              {connStatus === 'connected' && (
                <span className="text-xs text-emerald-400 flex items-center gap-1 font-mono">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Online
                </span>
              )}
              {connStatus === 'error' && (
                <span className="text-xs text-rose-400 flex items-center gap-1 font-mono">
                  <AlertCircle className="w-3.5 h-3.5" /> Disconnected
                </span>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <div className="space-y-1.5">
              <label className="text-slate-300 font-semibold text-xs block">
                Backend API Server Base URL (Laptop 1)
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="e.g. http://192.168.1.100:8000 or leave blank for local"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="flex-1 bg-background border border-border rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-primary transition"
                />
                <button
                  type="button"
                  onClick={() => testConnection(apiUrl)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-bold border border-border transition"
                >
                  Test Ping
                </button>
              </div>
              <p className="text-[11px] text-slate-400">
                {connMsg || "Leave blank when UI and FastAPI run on the same machine. Enter http://<Laptop1_IP>:8000 when running UI on Laptop 2."}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-2">
              <div className="p-3 bg-background rounded-lg border border-border space-y-1">
                <span className="text-slate-400">Host Binding</span>
                <p className="font-bold text-white font-mono">0.0.0.0:8000 (LAN)</p>
              </div>
              <div className="p-3 bg-background rounded-lg border border-border space-y-1">
                <span className="text-slate-400">CORS Policy</span>
                <p className="font-bold text-emerald-400 font-mono">Enabled (*)</p>
              </div>
              <div className="p-3 bg-background rounded-lg border border-border space-y-1">
                <span className="text-slate-400">Model Pipeline</span>
                <p className="font-bold text-primary font-mono">HMM + LightGBM</p>
              </div>
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary hover:bg-primary-hover text-white text-xs font-bold transition shadow-lg shadow-primary/20"
            >
              {saved ? (
                <>
                  <Check className="w-4 h-4" />
                  Saved Preferences!
                </>
              ) : (
                'Save Settings'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
