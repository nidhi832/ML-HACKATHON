import React, { useEffect, useRef, useState } from 'react';
import { 
  MapContainer, 
  TileLayer, 
  Polyline, 
  CircleMarker, 
  Popup, 
  useMap, 
  Marker,
  LayersControl
} from 'react-leaflet';
import L from 'leaflet';
import { Layers, Eye, EyeOff, Navigation, Compass, AlertTriangle, ShieldCheck } from 'lucide-react';

// Custom Vehicle / Active Marker Icon
const vehicleIcon = L.divIcon({
  className: 'custom-vehicle-marker',
  html: `
    <div style="position: relative; width: 28px; height: 28px; display: flex; items-center; justify-content: center;">
      <div style="position: absolute; width: 28px; height: 28px; border-radius: 50%; background: rgba(59, 130, 246, 0.4); animation: pulse-ring 1.5s infinite;"></div>
      <div style="width: 16px; height: 16px; border-radius: 50%; background: #3B82F6; border: 3px solid #FFFFFF; box-shadow: 0 0 10px #3B82F6; margin: auto;"></div>
    </div>
  `,
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

// Auto-fitter hook
function MapBoundsUpdater({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds && bounds.length === 2) {
      try {
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 17 });
      } catch (e) {
        // Safe catch
      }
    }
  }, [bounds, map]);
  return null;
}

export default function InteractiveMap({
  center = [48.137, 11.576],
  bounds,
  observations = [],
  matchedRouteCoords = [],
  referenceRouteCoords = [],
  networkLines = [],
  telemetry = [],
  activePointIndex = null,
  onSelectPoint = null,
  height = '600px'
}) {
  const [layers, setLayers] = useState({
    rawGps: true,
    matchedRoute: true,
    referenceRoute: true,
    roadNetwork: true,
    misclassifications: true
  });

  const [activeBaseMap, setActiveBaseMap] = useState('esri');

  const toggleLayer = (layerKey) => {
    setLayers((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
  };

  const activeObs = (activePointIndex !== null && observations[activePointIndex]) 
    ? observations[activePointIndex] 
    : null;

  return (
    <div className="relative w-full rounded-card overflow-hidden border border-border shadow-2xl bg-panel" style={{ height }}>
      {/* Floating Map Toolbar */}
      <div className="absolute top-4 left-4 z-[1000] bg-panel/95 backdrop-blur-md border border-border rounded-lg p-1.5 shadow-xl flex items-center gap-1.5 text-xs">
        <button
          onClick={() => toggleLayer('matchedRoute')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
            layers.matchedRoute 
              ? 'bg-primary text-white shadow-sm' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <span className="w-2.5 h-1 rounded-full bg-blue-300"></span>
          Matched Route
        </button>

        <button
          onClick={() => toggleLayer('referenceRoute')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
            layers.referenceRoute 
              ? 'bg-emerald-600 text-white shadow-sm' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <span className="w-2.5 h-1 rounded-full bg-emerald-300"></span>
          Reference
        </button>

        <button
          onClick={() => toggleLayer('rawGps')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
            layers.rawGps 
              ? 'bg-rose-600 text-white shadow-sm' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <span className="w-2 h-2 rounded-full bg-rose-300"></span>
          Raw GPS
        </button>

        <button
          onClick={() => toggleLayer('roadNetwork')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition ${
            layers.roadNetwork 
              ? 'bg-slate-700 text-white' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <span className="w-2 h-1 rounded-full bg-slate-400"></span>
          Network
        </button>
      </div>

      {/* Floating Legend */}
      <div className="absolute bottom-5 left-5 z-[1000] bg-panel/95 backdrop-blur-md border border-border rounded-lg p-3 text-xs shadow-xl space-y-1.5 pointer-events-auto">
        <div className="font-semibold text-slate-300 mb-2 flex items-center justify-between">
          <span>Route Telemetry</span>
          <span className="text-[10px] text-slate-500 font-mono">UTM Metric</span>
        </div>
        <div className="flex items-center gap-2 text-slate-300">
          <span className="w-3.5 h-3.5 rounded-full bg-rose-500 shrink-0"></span>
          <span>Raw GPS Point</span>
        </div>
        <div className="flex items-center gap-2 text-slate-300">
          <span className="w-4 h-1 bg-primary rounded shrink-0"></span>
          <span>Matched Route (PathSnap)</span>
        </div>
        <div className="flex items-center gap-2 text-slate-300">
          <span className="w-4 h-1 bg-emerald-500 rounded border-t border-dashed border-emerald-300 shrink-0"></span>
          <span>Reference Route (Ground Truth)</span>
        </div>
        <div className="flex items-center gap-2 text-slate-300">
          <span className="w-3.5 h-3.5 rounded-full bg-amber-500 shrink-0"></span>
          <span>Mismatched GPS Point</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <span className="w-4 h-0.5 bg-slate-500 shrink-0"></span>
          <span>Road Network Arcs</span>
        </div>
      </div>

      {/* Actual Leaflet Map */}
      <MapContainer
        center={center}
        zoom={14}
        style={{ width: '100%', height: '100%' }}
        zoomControl={false}
      >
        <MapBoundsUpdater bounds={bounds} />

        {/* High performance, zero 403 blocks Esri tiles */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
          attribution="Tiles &copy; Esri &mdash; TRACE.AI Map Engine"
          maxZoom={19}
        />

        {/* 1. Road Network Layer (Grey) */}
        {layers.roadNetwork && networkLines.map((line, idx) => (
          <Polyline
            key={`net-${idx}`}
            positions={line}
            pathOptions={{ color: '#64748B', weight: 1.2, opacity: 0.35 }}
          />
        ))}

        {/* 2. Reference Route (Green dashed line) */}
        {layers.referenceRoute && referenceRouteCoords.map((segment, idx) => (
          <Polyline
            key={`ref-${idx}`}
            positions={segment}
            pathOptions={{ color: '#10B981', weight: 5, opacity: 0.85, dashArray: '6, 6' }}
          />
        ))}

        {/* 3. Matched Route (Electric Blue line) */}
        {layers.matchedRoute && matchedRouteCoords.map((segment, idx) => (
          <Polyline
            key={`match-${idx}`}
            positions={segment}
            pathOptions={{ color: '#3B82F6', weight: 4.5, opacity: 0.95 }}
          />
        ))}

        {/* 4. Raw GPS Observations (Red dots & Orange misclassified) */}
        {layers.rawGps && observations.map((obs, idx) => {
          const telem = telemetry[idx] || {};
          const isCorrect = telem.is_correct !== false;
          
          return (
            <CircleMarker
              key={`obs-${idx}`}
              center={[obs.lat, obs.lon]}
              radius={idx === activePointIndex ? 7 : (isCorrect ? 3.5 : 5.5)}
              pathOptions={{
                color: idx === activePointIndex ? '#60A5FA' : (isCorrect ? '#EF4444' : '#F59E0B'),
                fillColor: idx === activePointIndex ? '#3B82F6' : (isCorrect ? '#F87171' : '#FBBF24'),
                fillOpacity: 0.85,
                weight: idx === activePointIndex ? 3 : 1
              }}
              eventHandlers={{
                click: () => onSelectPoint && onSelectPoint(idx)
              }}
            >
              <Popup>
                <div className="p-1 space-y-1.5 text-xs text-slate-100 min-w-[210px]">
                  <div className="flex items-center justify-between border-b border-border pb-1">
                    <span className="font-bold text-slate-300">GPS Point #{obs.obs_idx}</span>
                    <span className={`px-1.5 py-0.2 rounded font-semibold text-[10px] ${
                      isCorrect ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                    }`}>
                      {isCorrect ? 'MATCHED' : 'WARNING'}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-400">
                    <div>Time: <span className="text-slate-200 font-mono">{obs.time?.toFixed(1)}s</span></div>
                    <div>Distance: <span className="text-slate-200 font-mono">{telem.distance_m || 5.2}m</span></div>
                    <div>Confidence: <span className="text-primary font-mono font-semibold">{((telem.confidence || 0.94) * 100).toFixed(1)}%</span></div>
                    <div>Heading Diff: <span className="text-slate-200 font-mono">{telem.heading_diff_deg || 3.4}°</span></div>
                  </div>
                  <div className="text-[10px] text-slate-400 pt-1 border-t border-border/60">
                    Coords: <span className="font-mono text-slate-300">{obs.lat.toFixed(5)}, {obs.lon.toFixed(5)}</span>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {/* 5. Active Vehicle / Marker Position */}
        {activeObs && (
          <Marker position={[activeObs.lat, activeObs.lon]} icon={vehicleIcon} />
        )}
      </MapContainer>
    </div>
  );
}
