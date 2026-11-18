import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Search, 
  Filter, 
  ArrowUpDown, 
  Compass, 
  ChevronLeft, 
  ChevronRight, 
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  Clock
} from 'lucide-react';
import { fetchTrips } from '../services/api';

export default function TripExplorer({ activeDataset }) {
  const navigate = useNavigate();
  const [trips, setTrips] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState('accuracy_pct');
  const [sortAsc, setSortAsc] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  useEffect(() => {
    async function loadTrips() {
      const data = await fetchTrips(activeDataset);
      setTrips(data);
    }
    loadTrips();
  }, [activeDataset]);

  // Filtering
  const filteredTrips = trips.filter(t => 
    t.trip_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Sorting
  const sortedTrips = [...filteredTrips].sort((a, b) => {
    let valA = a[sortField];
    let valB = b[sortField];
    if (valA === null || valA === undefined) return 1;
    if (valB === null || valB === undefined) return -1;
    if (valA < valB) return sortAsc ? -1 : 1;
    if (valA > valB) return sortAsc ? 1 : -1;
    return 0;
  });

  // Pagination
  const totalPages = Math.ceil(sortedTrips.length / pageSize) || 1;
  const paginatedTrips = sortedTrips.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold text-white tracking-tight">Trip Explorer</h1>
          <span className="text-xs bg-primary/20 text-primary border border-primary/30 px-2 py-0.5 rounded-full font-semibold">
            {filteredTrips.length} Trajectories
          </span>
        </div>
        <p className="text-sm text-slate-400 mt-1">
          Search, filter, and inspect historical and evaluated GPS trajectories with ground-truth verification.
        </p>
      </div>

      {/* Filter & Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-panel border border-border p-4 rounded-card shadow-lg">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search Trip ID (e.g. 00000052)..."
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
            className="w-full bg-background border border-border rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-primary transition"
          />
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span>Dataset Partition: <strong className="text-white">{activeDataset}</strong></span>
          <span className="text-slate-600">•</span>
          <span>Showing {paginatedTrips.length} of {filteredTrips.length}</span>
        </div>
      </div>

      {/* Trips Table */}
      <div className="bg-panel border border-border rounded-card overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-background/80 border-b border-border text-slate-400 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3.5 px-4 cursor-pointer hover:text-white" onClick={() => handleSort('trip_id')}>
                  <div className="flex items-center gap-1.5">
                    Trip ID
                    <ArrowUpDown className="w-3.5 h-3.5" />
                  </div>
                </th>
                <th className="py-3.5 px-4 cursor-pointer hover:text-white" onClick={() => handleSort('gps_points')}>
                  <div className="flex items-center gap-1.5">
                    GPS Observations
                    <ArrowUpDown className="w-3.5 h-3.5" />
                  </div>
                </th>
                <th className="py-3.5 px-4 cursor-pointer hover:text-white" onClick={() => handleSort('distance_km')}>
                  <div className="flex items-center gap-1.5">
                    Distance (km)
                    <ArrowUpDown className="w-3.5 h-3.5" />
                  </div>
                </th>
                <th className="py-3.5 px-4">Duration</th>
                <th className="py-3.5 px-4">Matched Segments</th>
                <th className="py-3.5 px-4 cursor-pointer hover:text-white" onClick={() => handleSort('accuracy_pct')}>
                  <div className="flex items-center gap-1.5">
                    Matching Accuracy
                    <ArrowUpDown className="w-3.5 h-3.5" />
                  </div>
                </th>
                <th className="py-3.5 px-4">Status</th>
                <th className="py-3.5 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {paginatedTrips.map((t) => {
                const acc = t.accuracy_pct;
                return (
                  <tr key={t.trip_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-mono font-bold text-white flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-primary"></span>
                      {t.trip_id}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {t.gps_points.toLocaleString()} pts
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {t.distance_km} km
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400">
                      {t.duration_min} min
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400">
                      {t.route_segments} arcs
                    </td>
                    <td className="py-3 px-4">
                      {acc !== null ? (
                        <div className="flex items-center gap-2">
                          <span className={`font-mono font-bold ${
                            acc >= 90 ? 'text-emerald-400' : (acc >= 80 ? 'text-blue-400' : 'text-amber-400')
                          }`}>
                            {acc}%
                          </span>
                          <div className="w-16 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${
                                acc >= 90 ? 'bg-emerald-400' : (acc >= 80 ? 'bg-primary' : 'bg-amber-400')
                              }`}
                              style={{ width: `${acc}%` }}
                            ></div>
                          </div>
                        </div>
                      ) : (
                        <span className="text-slate-500 font-mono italic">No Label</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {t.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => navigate('/matching')}
                        className="p-1.5 rounded-lg bg-background border border-border text-slate-400 hover:text-primary hover:border-primary/50 transition"
                        title="Inspect in Map Matching"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="p-4 border-t border-border flex items-center justify-between text-xs text-slate-400 bg-background/40">
          <div>
            Page <span className="text-white font-semibold">{currentPage}</span> of <span className="text-white font-semibold">{totalPages}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage <= 1}
              className="p-1.5 rounded-lg bg-panel border border-border text-slate-300 hover:text-white disabled:opacity-40 transition"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="p-1.5 rounded-lg bg-panel border border-border text-slate-300 hover:text-white disabled:opacity-40 transition"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
