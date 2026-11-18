import React from 'react';

export default function MetricCard({ 
  title, 
  value, 
  subtitle, 
  badge, 
  icon: Icon, 
  color = 'primary' 
}) {
  const colorMap = {
    primary: {
      bg: 'bg-primary/10',
      border: 'border-primary/20',
      text: 'text-primary',
    },
    success: {
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
      text: 'text-emerald-400',
    },
    warning: {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20',
      text: 'text-amber-400',
    },
    error: {
      bg: 'bg-rose-500/10',
      border: 'border-rose-500/20',
      text: 'text-rose-400',
    }
  };

  const scheme = colorMap[color] || colorMap.primary;

  return (
    <div className="bg-panel border border-border rounded-card p-5 relative overflow-hidden transition-all duration-200 hover:border-slate-600 hover:shadow-lg hover:shadow-black/20">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 tracking-wide uppercase">{title}</p>
          <h3 className="text-2xl font-bold text-white mt-1 tracking-tight">{value}</h3>
        </div>
        {Icon && (
          <div className={`w-10 h-10 rounded-lg ${scheme.bg} ${scheme.border} border flex items-center justify-center`}>
            <Icon className={`w-5 h-5 ${scheme.text}`} />
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between text-xs">
        <span className="text-slate-500 font-medium">{subtitle}</span>
        {badge && (
          <span className={`px-2 py-0.5 rounded-full font-semibold text-[11px] ${scheme.bg} ${scheme.text} border ${scheme.border}`}>
            {badge}
          </span>
        )}
      </div>
    </div>
  );
}
