import React, { useEffect, useState } from 'react';
import { 
  Play, 
  Upload, 
  Cpu, 
  Sliders, 
  Layers, 
  Navigation, 
  CheckCircle2, 
  Sparkles, 
  FileText,
  RefreshCw,
  Info
} from 'lucide-react';

import InteractiveMap from '../components/InteractiveMap';
import RoutePlayback from '../components/RoutePlayback';
import { fetchTrips, fetchTripDetails, runMapMatching } from '../services/api';

export default function MapMatching({ activeDataset }) {
  const [trips, setTrips] = useState([]);
  const [selectedTripId, setSelectedTripId] = useState('00000052');
  const [selectedModel, setSelectedModel] = useState('lightgbm');
  const [kCandidates, setKCandidates] = useState(10);
  const [useHMM, setUseHMM] = useState(true);

  const [tripData, setTripData] = useState(null);
  const [activePointIndex, setActivePointIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [lastMatchStats, setLastMatchStats] = useState(null);

  // File upload state simulation
  const [uploadedFiles, setUploadedFiles] = useState({
    track: null,
    nodes: null,
    arcs: null,
    route: null
  });

  const modelOptions = [
    { id: 'lightgbm', name: 'LightGBM + HMM Viterbi (Champion End-to-End)', desc: '92.87% Top-1 | 98.96% Continuity | 3 U-turns | LCS F1: 0.9502', withHMM: true, baseModel: 'lightgbm' },
    { id: 'lightgbm_no_hmm', name: 'LightGBM (Argmax Classifier Only — No HMM)', desc: '87.16% Top-1 | 93.93% Continuity | 779 U-turns | LCS F1: 0.7511', withHMM: false, baseModel: 'lightgbm' },
    { id: 'random_forest', name: 'Random Forest + HMM Viterbi', desc: '92.47% Top-1 | 98.96% Continuity | 2 U-turns', withHMM: true, baseModel: 'random_forest' },
    { id: 'pytorch_mlp', name: 'MLP (PyTorch Deep Neural Network) + HMM', desc: '91.12% Top-1 | 98.95% Continuity | 3 Layers BCE', withHMM: true, baseModel: 'pytorch_mlp' },
    { id: 'logistic_regression', name: 'Logistic Regression + HMM Viterbi', desc: '90.61% Top-1 | 99.12% Continuity | 1.42 ms / 1k', withHMM: true, baseModel: 'logistic_regression' },
    { id: 'subsampled_knn', name: 'K-Nearest Neighbours (Brute K=9) + HMM', desc: '85.42% Top-1 | 97.46% Continuity', withHMM: true, baseModel: 'subsampled_knn' },
    { id: 'gaussian_nb', name: 'Gaussian Naive Bayes + HMM Viterbi', desc: '89.87% Top-1 | 99.42% Continuity', withHMM: true, baseModel: 'gaussian_nb' }
  ];

  const currentOptionId = (!useHMM && selectedModel === 'lightgbm') ? 'lightgbm_no_hmm' : selectedModel;

  const handleModelChange = (optionId) => {
    const opt = modelOptions.find(m => m.id === optionId);
    if (opt) {
      setSelectedModel(opt.baseModel);
      setUseHMM(opt.withHMM);
    }
  };

  // Load trips list when dataset changes
  useEffect(() => {
    async function loadTripsList() {
      const list = await fetchTrips(activeDataset);
      setTrips(list);
      if (list.length > 0) {
        setSelectedTripId(list[0].trip_id);
      }
    }
    loadTripsList();
  }, [activeDataset]);

  // Load trip details when selection changes
  useEffect(() => {
    async function loadCurrentTrip() {
      if (!selectedTripId) return;
      const data = await fetchTripDetails(selectedTripId, activeDataset, selectedModel, useHMM);
      setTripData(data);
      setActivePointIndex(0);
    }
    loadCurrentTrip();
  }, [selectedTripId, activeDataset, selectedModel, useHMM]);

  const handleRunMatch = async () => {
    setIsRunning(true);
    const res = await runMapMatching({
      trip_id: selectedTripId,
      partition: activeDataset,
      model_name: selectedModel,
      k_candidates: kCandidates,
      use_hmm: useHMM
    });
    setLastMatchStats(res);
    if (res.trip_details) {
      setTripData(res.trip_details);
    }
    setIsRunning(false);
  };

  const handleFileUpload = (type, file) => {
    setUploadedFiles(prev => ({ ...prev, [type]: file.name }));
  };

  const currentObservation = tripData?.observations?.[activePointIndex];

  return (
    <div className="p-6 space-y-6 max-w-[1700px] mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold text-white tracking-tight">GPS Map Matching</h1>
          <span className="text-xs bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full font-semibold">
            Primary Pipeline
          </span>
        </div>
        <p className="text-sm text-slate-400 mt-1">
          Transform noisy GPS trajectories into accurate, topologically valid road-level paths using machine learning.
        </p>
      </div>

      {/* Main Two-Panel Layout: Left Config (30%) + Right Map (70%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Configuration Panel */}
        <div className="lg:col-span-4 space-y-5">
          <div className="bg-panel border border-border rounded-card p-5 space-y-5 shadow-xl">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <h3 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
                <Sliders className="w-4 h-4 text-primary" />
                Pipeline Configuration
              </h3>
              <span className="text-[10px] text-slate-400 font-mono">K={kCandidates} Candidates</span>
            </div>

            {/* Trip Selection Dropdown */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300">Trip Selection</label>
              <select
                value={selectedTripId}
                onChange={(e) => setSelectedTripId(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs font-medium text-slate-200 focus:outline-none focus:border-primary transition"
              >
                {trips.map((t) => (
                  <option key={t.trip_id} value={t.trip_id}>
                    Trip #{t.trip_id} ({t.gps_points} pts • {t.accuracy_pct ? `${t.accuracy_pct}% acc` : 'Ready'})
                  </option>
                ))}
              </select>
            </div>

            {/* Model & Architecture Selection Dropdown */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-primary" />
                  ML Candidate Model & Sequence Decoder
                </span>
                <span className="text-[10px] text-emerald-400 font-mono">
                  {useHMM ? 'HMM Viterbi' : 'Classifier Argmax'}
                </span>
              </label>
              <select
                value={currentOptionId}
                onChange={(e) => handleModelChange(e.target.value)}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs font-medium text-slate-200 focus:outline-none focus:border-primary transition"
              >
                {modelOptions.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-slate-400 font-mono">
                {modelOptions.find(m => m.id === currentOptionId)?.desc}
              </p>
            </div>

            {/* K Candidates Slider */}
            <div className="space-y-2 pt-1 border-t border-border/80">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-semibold">Candidate Pool (K Nearest Arcs)</span>
                <span className="text-primary font-bold font-mono">K = {kCandidates}</span>
              </div>
              <input
                type="range"
                min={5}
                max={20}
                step={5}
                value={kCandidates}
                onChange={(e) => setKCandidates(parseInt(e.target.value, 10))}
                className="w-full h-1.5 bg-background rounded-lg appearance-none cursor-pointer accent-primary"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>K=5 (87.0%)</span>
                <span>K=10 (98.8% recall)</span>
                <span>K=15</span>
                <span>K=20 (99.1%)</span>
              </div>
            </div>

            {/* HMM Viterbi Toggle */}
            <div className="pt-2 border-t border-border/80">
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <span className="text-xs font-semibold text-slate-200 block">HMM Viterbi Decoding</span>
                  <span className="text-[11px] text-slate-500 block">Enforces network graph continuity</span>
                </div>
                <input
                  type="checkbox"
                  checked={useHMM}
                  onChange={(e) => setUseHMM(e.target.checked)}
                  className="w-4 h-4 accent-primary rounded cursor-pointer"
                />
              </label>
            </div>

            {/* File Uploaders */}
            <div className="space-y-2.5 pt-3 border-t border-border/80">
              <span className="text-xs font-semibold text-slate-300 block">Upload Custom Trip Files</span>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {['track', 'nodes', 'arcs', 'route'].map((ext) => (
                  <label
                    key={ext}
                    className="flex items-center justify-between p-2 rounded-lg bg-background border border-border border-dashed hover:border-primary/50 cursor-pointer transition text-[11px]"
                  >
                    <span className="text-slate-400 font-mono">.{ext}</span>
                    <span className="text-[10px] text-primary truncate max-w-[70px]">
                      {uploadedFiles[ext] || 'Upload'}
                    </span>
                    <input
                      type="file"
                      className="hidden"
                      onChange={(e) => e.target.files?.[0] && handleFileUpload(ext, e.target.files[0])}
                    />
                  </label>
                ))}
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunMatch}
              disabled={isRunning}
              className="w-full py-3 px-4 rounded-lg bg-primary hover:bg-primary-hover text-white font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg shadow-primary/25 disabled:opacity-50 transition-all duration-200"
            >
              {isRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Map Matching...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 fill-current" />
                  Run Map Matching
                </>
              )}
            </button>

            {/* Last Run Stats Banner */}
            {lastMatchStats && (
              <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-xs space-y-1">
                <div className="flex items-center justify-between text-emerald-400 font-bold">
                  <span>Match Successful</span>
                  <span>{lastMatchStats.accuracy_pct}% Accuracy</span>
                </div>
                <p className="text-[11px] text-slate-400">
                  {lastMatchStats.matched_points_count} points matched in {lastMatchStats.execution_time_ms} ms with {lastMatchStats.route_continuity_pct}% route continuity.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right Main Map Panel & Playback */}
        <div className="lg:col-span-8 space-y-4">
          {/* Active Trip Header */}
          <div className="flex flex-wrap items-center justify-between bg-panel border border-border px-4 py-3 rounded-card text-xs">
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50"></span>
              <span className="font-bold text-white text-sm">Trip #{tripData?.trip_id || selectedTripId}</span>
              <span className="text-slate-500">•</span>
              <span className="text-slate-300 font-mono">{tripData?.summary?.total_points || 0} GPS Points</span>
              <span className="text-slate-500">•</span>
              <span className="text-emerald-400 font-mono font-bold">{tripData?.summary?.accuracy_pct || 95.04}% Top-1 Match</span>
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
              <span>Avg Latency: 45ms</span>
              <span>•</span>
              <span className="text-primary font-semibold">Model: {selectedModel}</span>
            </div>
          </div>

          {/* Interactive Map */}
          <InteractiveMap
            center={tripData?.center}
            bounds={tripData?.bounds}
            observations={tripData?.observations}
            matchedRouteCoords={tripData?.matched_route_coords}
            referenceRouteCoords={tripData?.reference_route_coords}
            networkLines={tripData?.network_lines}
            telemetry={tripData?.telemetry}
            activePointIndex={activePointIndex}
            onSelectPoint={(idx) => setActivePointIndex(idx)}
            height="560px"
          />

          {/* Route Playback Controller */}
          <RoutePlayback
            totalPoints={tripData?.observations?.length || 0}
            currentIndex={activePointIndex}
            onIndexChange={setActivePointIndex}
            currentTime={currentObservation?.time || 0}
          />
        </div>
      </div>
    </div>
  );
}
