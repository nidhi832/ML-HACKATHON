import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import Dashboard from './pages/Dashboard';
import MapMatching from './pages/MapMatching';
import TripExplorer from './pages/TripExplorer';
import ModelComparison from './pages/ModelComparison';
import EvaluationDashboard from './pages/EvaluationDashboard';
import Settings from './pages/Settings';

export default function App() {
  const [activeDataset, setActiveDataset] = useState('student_test');

  return (
    <BrowserRouter>
      <div className="flex h-screen w-screen overflow-hidden bg-background text-slate-100 antialiased">
        {/* Left Fixed Collapsible Sidebar */}
        <Sidebar />

        {/* Right Main Application Area */}
        <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
          <Topbar 
            activeDataset={activeDataset} 
            setActiveDataset={setActiveDataset} 
          />

          {/* Main Scrollable Content */}
          <main className="flex-1 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Dashboard activeDataset={activeDataset} />} />
              <Route path="/matching" element={<MapMatching activeDataset={activeDataset} />} />
              <Route path="/trips" element={<TripExplorer activeDataset={activeDataset} />} />
              <Route path="/models" element={<ModelComparison />} />
              <Route path="/evaluation" element={<EvaluationDashboard />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
