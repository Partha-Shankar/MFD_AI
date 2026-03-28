import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Terminal, RefreshCw, AlertCircle } from 'lucide-react';

const SystemLogs = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const endOfLogsRef = useRef<HTMLDivElement>(null);

  const fetchLogs = async () => {
    try {
      // The image analysis engine runs on port 8002
      const response = await fetch('http://localhost:8002/system-logs');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setLogs(data.logs || []);
      setError(null);
    } catch (err: any) {
      console.error('Error fetching system logs:', err);
      setError('Failed to connect to the backend engine to fetch logs.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000); // Polling every 2s
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Auto scroll to bottom
    if (endOfLogsRef.current) {
      endOfLogsRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-8 max-w-6xl mx-auto h-[calc(100vh-80px)] flex flex-col"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-heading font-bold text-gray-900 flex items-center gap-3">
            <Terminal className="text-primary-600" />
            System Backend Logs
          </h1>
          <p className="text-gray-500 mt-2">
            Real-time execution logs from the core Python Image Forensics engine.
          </p>
        </div>
        <button
          onClick={() => fetchLogs()}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700 shadow-sm"
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 text-red-700 rounded-xl flex items-center gap-3">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      <div className="flex-1 bg-[#1e1e1e] rounded-xl overflow-hidden shadow-2xl border border-gray-800 flex flex-col">
        <div className="h-10 bg-[#2d2d2d] border-b border-gray-800 flex items-center px-4 gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span className="ml-4 text-xs font-mono text-gray-400">image_analysis/system_backend.log</span>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 font-mono text-sm">
          {logs.length === 0 && !loading && !error ? (
            <div className="text-gray-500 italic">No logs available.</div>
          ) : (
            logs.map((log, index) => {
              // Basic syntax highlighting for log levels
              let colorClass = "text-gray-300";
              if (log.includes("[ERROR]") || log.includes("Error:") || log.includes("Exception:")) {
                colorClass = "text-red-400";
              } else if (log.includes("[WARNING]")) {
                colorClass = "text-yellow-400";
              } else if (log.includes("[INFO]")) {
                colorClass = "text-blue-400";
              } else if (log.includes("loaded") || log.includes("success")) {
                colorClass = "text-green-400";
              }

              return (
                <div key={index} className={`whitespace-pre-wrap break-all mb-1 ${colorClass}`}>
                  {log}
                </div>
              );
            })
          )}
          <div ref={endOfLogsRef} />
        </div>
      </div>
    </motion.div>
  );
};

export default SystemLogs;
