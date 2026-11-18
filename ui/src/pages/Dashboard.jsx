import React, { useEffect, useState } from 'react';
import { 
  Navigation, 
  MapPin, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  Activity, 
  Zap,
  TrendingUp,
  Crosshair
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  Tooltip 
} from 'recharts';

import MetricCard from '../components/MetricCard';
import InteractiveMap from '../components/InteractiveMap';
import { fetchOverview, fetchTripDetails } from '../services/api';

export default function Dashboard({ activeDataset }) {
  const [overview, setOverview] = useState(null);
  const [tripData, setTripData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      const ov = await fetchOverview();
      setOverview(ov);
      const trip = await fetchTripDetails(ov.featured_trip_id || "00000052", activeDataset, "lightgbm");
      setTripData(trip);
      setLoading(false);
    }
    loadData();
  }, [activeDataset]);

  // Confidence distribution data for mini-chart
  const confidenceData = [
    { range: '0-50%', count: 4 },
    { range: '50-70%', count: 8 },
    { range: '70-85%', count: 22 },
    { range: '85-95%', count: 94 },
    { range: '95-100%', count: 134 },
  ];

  const kpis = overview?.kpis || {
    total_trips: 90,
    total_gps_points: 227577,
    top1_accuracy_pct: 87.16,
    candidate_miss_rate_pct: 1.17,
    avg_processing_time_ms: 45.01,
  };

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold text-white tracking-tight">Route Intelligence Overview</h1>
          <span className="text-xs bg-primary/20 text-primary border border-primary/30 px-2 py-0.5 rounded-full font-semibold">
            Live Telemetry
          </span>
        </div>
        <p className="text-sm text-slate-400 mt-1">
          Monitor GPS trajectories, matching performance and model predictions across directed road networks.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Total Trips"
          value={kpis.total_trips.toLocaleString()}
          subtitle="80 Train / 10 Test"
          badge="Complete"
          icon={Navigation}
          color="primary"
        />
        <MetricCard
          title="Total GPS Points"
          value={kpis.total_gps_points.toLocaleString()}
          subtitle="94.3% sampled at 1.0s"
          badge="Verified"
          icon={Activity}
          color="primary"
        />
        <MetricCard
          title="Map Matching Accuracy"
          value={`${kpis.top1_accuracy_pct}%`}
          subtitle="LightGBM Champion"
          badge="+3.0% Tuned"
          icon={CheckCircle2}
          color="success"
        />
        <MetricCard
          title="Candidate Miss Rate"
          value={`${kpis.candidate_miss_rate_pct}%`}
          subtitle="98.83% Spatial Recall"
          badge="K=10"
          icon={Crosshair}
          color="success"
        />
        <MetricCard
          title="Avg Processing Time"
          value={`${kpis.avg_processing_time_ms} ms`}
          subtitle="Per 1,000 observations"
          badge="Realtime"
          icon={Zap}
          color="warning"
        />
      </div>

      {/* Main Grid: Interactive Map (68%) + Right Analytics Panel (32%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Map Area */}
        <div className="lg:col-span-8 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse"></span>
              <h2 className="text-sm font-semibold text-slate-200">
                Featured Trajectory: Trip #{tripData?.trip_id || '00000052'}
              </h2>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              Esri WorldStreetMap • EPSG Metric UTM
            </span>
          </div>

          <InteractiveMap
            center={tripData?.center}
            bounds={tripData?.bounds}
            observations={tripData?.observations}
            matchedRouteCoords={tripData?.matched_route_coords}
            referenceRouteCoords={tripData?.reference_route_coords}
            networkLines={tripData?.network_lines}
            telemetry={tripData?.telemetry}
            height="580px"
          />
        </div>

        {/* Right Analytics Panel */}
        <div className="lg:col-span-4 space-y-5">
          <div className="bg-panel border border-border rounded-card p-5 space-y-5 shadow-xl">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <h3 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
                <Activity className="w-4 h-4 text-primary" />
                Active Trip Telemetry
              </h3>
              <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                Trip #{tripData?.trip_id || '00000052'}
              </span>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-background/80 border border-border/80 p-3 rounded-lg">
                <span className="text-slate-400">GPS Observations</span>
                <p className="text-lg font-bold text-white mt-0.5">{tripData?.summary?.total_points || 262}</p>
              </div>
              <div className="bg-background/80 border border-border/80 p-3 rounded-lg">
                <span className="text-slate-400">Matched Segments</span>
                <p className="text-lg font-bold text-white mt-0.5">{tripData?.summary?.matched_segments || 48}</p>
              </div>
              <div className="bg-background/80 border border-border/80 p-3 rounded-lg">
                <span className="text-slate-400">Avg Road Distance</span>
                <p className="text-lg font-bold text-white mt-0.5">{tripData?.summary?.avg_distance_m || 6.84} m</p>
              </div>
              <div className="bg-background/80 border border-border/80 p-3 rounded-lg">
                <span className="text-slate-400">Route Continuity</span>
                <p className="text-lg font-bold text-emerald-400 mt-0.5">{tripData?.summary?.route_continuity_pct || 98.5}%</p>
              </div>
            </div>

            {/* Matching Accuracy Bar */}
            <div className="space-y-1.5 pt-1">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-slate-400">Top-1 Matching Confidence</span>
                <span className="text-primary font-bold font-mono">
                  {tripData?.summary?.accuracy_pct || 95.04}%
                </span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div
                  className="bg-gradient-to-r from-primary to-emerald-400 h-full rounded-full transition-all duration-500"
                  style={{ width: `${tripData?.summary?.accuracy_pct || 95.04}%` }}
                ></div>
              </div>
            </div>

            {/* Confidence Distribution Chart */}
            <div className="pt-2">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-300">Confidence Distribution</span>
                <span className="text-[10px] text-slate-500 font-mono">P(candidate | obs)</span>
              </div>
              <div className="h-32 w-full bg-background/50 rounded-lg p-2 border border-border/60">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={confidenceData}>
                    <defs>
                      <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.6}/>
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.0}/>
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="range" stroke="#64748B" fontSize={10} tickLine={false} />
                    <YAxis stroke="#64748B" fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#111827', borderColor: '#253247', fontSize: '11px', borderRadius: '8px' }}
                      itemStyle={{ color: '#60A5FA' }}
                    />
                    <Area type="monotone" dataKey="count" stroke="#3B82F6" strokeWidth={2} fillOpacity={1} fill="url(#colorCount)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Pipeline Stage Footnote */}
            <div className="p-3 bg-primary/5 border border-primary/15 rounded-lg flex items-center gap-2 text-xs text-slate-300">
              <TrendingUp className="w-4 h-4 text-primary shrink-0" />
              <span>HMM Viterbi sequence decoding guarantees continuous connected road graph traversal.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
