import React from 'react';
import { useLocation } from 'react-router-dom';
import { Bell, ChevronRight, Database, Globe2, ShieldCheck, User } from 'lucide-react';

export default function Topbar({ activeDataset, setActiveDataset }) {
  const location = useLocation();

  const getPageTitle = (path) => {
    switch (path) {
      case '/': return 'Overview Dashboard';
      case '/matching': return 'GPS Map Matching';
      case '/trips': return 'Trip Explorer';
      case '/models': return 'Model Performance';
      case '/evaluation': return 'Evaluation Benchmark';
      case '/settings': return 'Platform Settings';
      default: return 'Route Intelligence';
    }
  };

  return (
    <header className="h-16 bg-panel/80 backdrop-blur border-b border-border px-6 flex items-center justify-between z-20 shrink-0 sticky top-0">
      {/* Breadcrumb & Project Name */}
      <div className="flex items-center gap-3 text-sm">
        <span className="font-semibold text-slate-400">PathSnap</span>
        <ChevronRight className="w-4 h-4 text-slate-600" />
        <span className="font-semibold text-slate-200">{getPageTitle(location.pathname)}</span>
        <span className="hidden md:inline-flex text-[11px] font-mono bg-panel-light text-slate-400 px-2 py-0.5 rounded border border-border">
          UE24CS352A
        </span>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-4">
        {/* Dataset Selector */}
        <div className="flex items-center bg-background border border-border rounded-lg p-1 text-xs font-medium">
          <Database className="w-3.5 h-3.5 text-primary ml-2 mr-1.5" />
          <button
            onClick={() => setActiveDataset && setActiveDataset('student_test')}
            className={`px-2.5 py-1 rounded transition-all ${
              activeDataset === 'student_test'
                ? 'bg-primary text-white shadow-sm font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            student_test (10)
          </button>
          <button
            onClick={() => setActiveDataset && setActiveDataset('train')}
            className={`px-2.5 py-1 rounded transition-all ${
              activeDataset === 'train'
                ? 'bg-primary text-white shadow-sm font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            train (80)
          </button>
        </div>

        {/* Global CRS Indicator */}
        <div className="hidden lg:flex items-center gap-1.5 text-xs text-slate-400 bg-background/50 border border-border px-2.5 py-1.5 rounded-lg">
          <Globe2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Metric UTM Auto</span>
        </div>

        {/* Notifications */}
        <button className="relative p-2 rounded-lg bg-background border border-border text-slate-400 hover:text-white hover:bg-slate-800 transition">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-primary"></span>
        </button>

        {/* System Status Pill */}
        <div className="flex items-center gap-2 pl-2 border-l border-border">
          <div className="w-2.5 h-2.5 rounded-full bg-success"></div>
          <span className="text-xs font-semibold text-slate-300 hidden sm:inline">Engine Ready</span>
        </div>
      </div>
    </header>
  );
}
