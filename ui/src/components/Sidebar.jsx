import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Compass, 
  MapPin, 
  Navigation, 
  Activity, 
  BarChart3, 
  ShieldCheck, 
  Settings, 
  Cpu, 
  CheckCircle2,
  Layers,
  Sparkles
} from 'lucide-react';

export default function Sidebar({ isOpen, setIsOpen }) {
  const navItems = [
    { name: 'Overview', path: '/', icon: Activity },
    { name: 'Map Matching', path: '/matching', icon: Navigation, badge: 'Core' },
    { name: 'Trip Explorer', path: '/trips', icon: Compass },
    { name: 'Model Performance', path: '/models', icon: BarChart3 },
    { name: 'Evaluation', path: '/evaluation', icon: ShieldCheck },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-panel border-r border-border flex flex-col justify-between shrink-0 h-screen select-none z-30 transition-all duration-300">
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center px-6 border-b border-border gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-primary to-blue-400 flex items-center justify-center shadow-lg shadow-primary/20">
            <Navigation className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-lg tracking-tight text-white">TRACE<span className="text-primary">.AI</span></span>
              <span className="text-[10px] font-semibold bg-primary/20 text-primary border border-primary/30 px-1.5 py-0.2 rounded">PRO</span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">Route Intelligence</p>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="px-3 py-6 space-y-1">
          <div className="px-3 pb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Platform
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-primary/10 text-primary border border-primary/20 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`
                }
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </div>
                {item.badge && (
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-primary text-white">
                    {item.badge}
                  </span>
                )}
              </NavLink>
            );
          })}
        </div>
      </div>

      {/* Bottom Status & User */}
      <div className="p-4 border-t border-border space-y-4">
        {/* System Telemetry Widget */}
        <div className="bg-background/80 rounded-lg p-3 border border-border/80 text-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
              Core System
            </span>
            <span className="text-slate-200 font-semibold">Online</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-primary" />
              Active Model
            </span>
            <span className="text-primary font-medium">LightGBM v2</span>
          </div>
        </div>

        {/* User Profile */}
        <div className="flex items-center gap-3 pt-1">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-slate-700 to-slate-600 border border-slate-500 flex items-center justify-center font-bold text-xs text-white">
            PS
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-200 truncate">PathSnap Team</p>
            <p className="text-[10px] text-slate-500 truncate">PES ML Hackathon</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
