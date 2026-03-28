import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface Stage {
  agent: string;
  message: string;
}

interface AnalysisLogsProps {
  stages: Stage[];
  progress: number;
  stageIndex: number;
  fileName?: string;
}

const AnalysisLogs: React.FC<AnalysisLogsProps> = ({ stages, progress, stageIndex, fileName }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [stageIndex]);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.97 }}
      className="bg-gray-950 rounded-[2rem] p-10 text-white relative overflow-hidden grid md:grid-cols-2 gap-10"
    >
      {/* Animated background grid */}
      <div className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: 'linear-gradient(rgba(139,92,246,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.5) 1px, transparent 1px)',
          backgroundSize: '40px 40px'
        }}
      />
      {/* Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-32 bg-violet-600 opacity-20 blur-3xl rounded-full pointer-events-none" />

      {/* Left Column: Modules */}
      <div className="relative z-10 flex flex-col h-[400px]">
        <div className="flex items-center gap-3 mb-8">
          <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
            <div className="w-3 h-3 rounded-full bg-yellow-500 animate-pulse" style={{ animationDelay: '0.2s' }} />
            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" style={{ animationDelay: '0.4s' }} />
          </div>
          <span className="text-gray-400 text-xs font-mono font-bold tracking-widest">ACTIVE FORENSIC MODULES</span>
        </div>

        <div className="flex-1 overflow-y-auto pr-4 space-y-4 font-mono text-sm custom-scrollbar">
          {stages.slice(0, stageIndex + 1).map((stage, i) => {
            const isActive = i === stageIndex;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`flex items-center gap-3 p-3 rounded-xl border ${isActive ? 'bg-violet-900/30 border-violet-500/50 text-violet-300' : 'bg-gray-900 border-gray-800 text-gray-500'}`}
              >
                <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-violet-400 animate-pulse' : 'bg-green-500'}`} />
                <span className="font-bold flex-1">{stage.agent}</span>
                {!isActive && <span className="text-[10px] text-green-500 border border-green-500/30 px-2 py-0.5 rounded-full">IDLE</span>}
                {isActive && <span className="text-[10px] text-violet-400 border border-violet-500/30 px-2 py-0.5 rounded-full animate-pulse">RUNNING</span>}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Right Column: Logs */}
      <div className="relative z-10 flex flex-col h-[400px] bg-black/50 rounded-2xl border border-gray-800 p-6">
        <div className="flex items-center justify-between mb-4 border-b border-gray-800 pb-4">
          <span className="text-gray-400 text-xs font-mono font-bold tracking-widest">LIVE SYSTEM LOGS</span>
          <span className="text-violet-400 text-xs font-mono">{Math.round(progress)}%</span>
        </div>
        
        <div ref={containerRef} className="flex-1 overflow-y-auto space-y-3 font-mono text-xs custom-scrollbar">
          {stages.slice(0, stageIndex + 1).map((stage, i) => {
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-gray-300 leading-relaxed"
              >
                <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span>{' '}
                <span className="text-violet-400 font-bold">[{stage.agent}]</span>{' '}
                {stage.message}
              </motion.div>
            );
          })}
        </div>

        {/* Progress bar */}
        <div className="mt-6 space-y-2">
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-violet-600 to-purple-400"
              style={{ width: `${progress}%` }}
              transition={{ ease: 'linear' }}
            />
          </div>
          {fileName && (
            <p className="text-gray-500 text-[10px] font-mono animate-pulse truncate">
              TARGET: {fileName}
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default AnalysisLogs;
