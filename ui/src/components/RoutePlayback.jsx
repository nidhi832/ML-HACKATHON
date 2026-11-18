import React, { useEffect, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, FastForward, Clock } from 'lucide-react';

export default function RoutePlayback({
  totalPoints = 0,
  currentIndex = 0,
  onIndexChange,
  currentTime = 0,
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1); // 1x, 2x, 4x

  useEffect(() => {
    let interval = null;
    if (isPlaying && totalPoints > 0) {
      interval = setInterval(() => {
        onIndexChange((prev) => {
          if (prev >= totalPoints - 1) {
            setIsPlaying(false);
            return 0; // loop or reset
          }
          return prev + 1;
        });
      }, 1000 / playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, totalPoints, onIndexChange]);

  const handleSliderChange = (e) => {
    onIndexChange(parseInt(e.target.value, 10));
  };

  const handlePrev = () => {
    onIndexChange((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    onIndexChange((prev) => Math.min(totalPoints - 1, prev + 1));
  };

  return (
    <div className="bg-panel border border-border rounded-card p-4 flex flex-col md:flex-row items-center justify-between gap-4 select-none shadow-xl">
      {/* Play Controls & Speed */}
      <div className="flex items-center gap-2">
        <button
          onClick={handlePrev}
          disabled={currentIndex <= 0}
          className="p-2 rounded-lg bg-background border border-border text-slate-300 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition"
          title="Previous Point"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`p-2.5 rounded-lg font-semibold flex items-center justify-center transition shadow-md ${
            isPlaying
              ? 'bg-amber-500 text-slate-950 hover:bg-amber-400'
              : 'bg-primary text-white hover:bg-primary-hover'
          }`}
          title={isPlaying ? 'Pause' : 'Play Trajectory'}
        >
          {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
        </button>

        <button
          onClick={handleNext}
          disabled={currentIndex >= totalPoints - 1}
          className="p-2 rounded-lg bg-background border border-border text-slate-300 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition"
          title="Next Point"
        >
          <SkipForward className="w-4 h-4" />
        </button>

        {/* Speed Selector */}
        <div className="flex items-center bg-background border border-border rounded-lg p-0.5 ml-2 text-xs">
          {[1, 2, 4].map((speed) => (
            <button
              key={speed}
              onClick={() => setPlaybackSpeed(speed)}
              className={`px-2 py-1 rounded font-mono font-medium transition ${
                playbackSpeed === speed
                  ? 'bg-primary text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {/* Timeline Slider & Scrubber */}
      <div className="flex-1 w-full flex items-center gap-4">
        <span className="text-xs font-mono text-slate-400 shrink-0 w-12 text-right">
          #{currentIndex}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, totalPoints - 1)}
          value={currentIndex}
          onChange={handleSliderChange}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <span className="text-xs font-mono text-slate-500 shrink-0 w-12">
          #{Math.max(0, totalPoints - 1)}
        </span>
      </div>

      {/* Timestamp Display */}
      <div className="flex items-center gap-2 bg-background/80 border border-border px-3 py-1.5 rounded-lg text-xs font-mono text-slate-300">
        <Clock className="w-3.5 h-3.5 text-primary" />
        <span>Time: {currentTime?.toFixed(1) || '0.0'}s</span>
      </div>
    </div>
  );
}
