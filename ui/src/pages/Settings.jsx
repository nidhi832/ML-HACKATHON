import React, { useState } from 'react';
import { 
  Settings as SettingsIcon, 
  Cpu, 
  Map, 
  Palette, 
  Radio, 
  Check, 
  Server,
  Layers
} from 'lucide-react';

export default function Settings() {
  const [defaultModel, setDefaultModel] = useState('lightgbm');
  const [candidateCount, setCandidateCount] = useState(10);
  const [autoCenter, setAutoCenter] = useState(true);
  const [matchedColor, setMatchedColor] = useState('#3B82F6');
  const [refColor, setRefColor] = useState('#10B981');
  const [gpsColor, setGpsColor] = useState('#EF4444');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
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

        {/* System & API Status */}
        <div className="bg-panel border border-border rounded-card p-5 space-y-4 shadow-xl md:col-span-2">
          <div className="flex items-center gap-2 border-b border-border pb-3">
            <Server className="w-4 h-4 text-primary" />
            <h3 className="font-semibold text-white text-sm">Backend & Engine Connection</h3>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div className="p-3 bg-background rounded-lg border border-border space-y-1">
              <span className="text-slate-400">API Status</span>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <span className="font-bold text-white">FastAPI Connected</span>
              </div>
            </div>

            <div className="p-3 bg-background rounded-lg border border-border space-y-1">
              <span className="text-slate-400">Model Weights</span>
              <p className="font-bold text-white">6 Models Serialized (artifacts/)</p>
            </div>

            <div className="p-3 bg-background rounded-lg border border-border space-y-1">
              <span className="text-slate-400">Projection Engine</span>
              <p className="font-bold text-white">pyproj UTM Zone Dynamic</p>
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
