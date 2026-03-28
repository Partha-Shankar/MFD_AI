import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Link2,
  Search,
  AlertTriangle,
  ShieldX,
  ShieldCheck,
  ChevronRight,
  RotateCcw,
  Youtube,
  Instagram,
  Cpu,
  Scan,
  Eye,
  Activity,
  Zap,
  Database,
  FileSearch,
  Brain,
  CheckCircle2,
} from 'lucide-react';
import api from '../services/api';
import AnalysisLogs, { Stage } from '../components/AnalysisLogs';

const ANALYSIS_STAGES: Stage[] = [
  { agent: 'Forensic Engine', message: 'Initializing forensic engine...' },
  { agent: 'Metadata Extractor', message: 'Collecting media metadata...' },
  { agent: 'Media Parser', message: 'Extracting video frames...' },
  { agent: 'Vision AI', message: 'Analyzing frame-level artifacts...' },
  { agent: 'GAN Detector', message: 'Detecting GAN diffusion patterns...' },
  { agent: 'Motion Tracker', message: 'Tracing motion consistency...' },
  { agent: 'Compression Inspector', message: 'Scanning compression signatures...' },
  { agent: 'Semantic AI', message: 'Evaluating semantic authenticity...' },
  { agent: 'Database Indexer', message: 'Cross-referencing deepfake datasets...' },
  { agent: 'Report Generator', message: 'Generating authenticity report...' },
];

const LinkAnalysis = () => {
  const [url, setUrl] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const intervalRef = useRef<any>(null);
  const stageIntervalRef = useRef<any>(null);

  const isValidUrl = (val: string) => {
    const lower = val.toLowerCase();
    return (
      lower.includes('youtube.com') ||
      lower.includes('youtu.be') ||
      lower.includes('instagram.com')
    );
  };

  const detectPlatformIcon = (val: string) => {
    const lower = val.toLowerCase();
    if (lower.includes('youtube.com') || lower.includes('youtu.be')) return Youtube;
    if (lower.includes('instagram.com')) return Instagram;
    return Link2;
  };

  const handleAnalyze = async () => {
    if (!url.trim()) {
      setError('Please paste a YouTube or Instagram URL.');
      return;
    }
    if (!isValidUrl(url)) {
      setError('Only YouTube and Instagram links are supported.');
      return;
    }
    setError('');
    setResult(null);
    setAnalyzing(true);
    setProgress(0);
    setStageIndex(0);

    // Progress bar over 45 seconds
    const totalMs = 45000;
    const tickMs = 200;
    let elapsed = 0;
    intervalRef.current = setInterval(() => {
      elapsed += tickMs;
      const pct = Math.min((elapsed / totalMs) * 100, 98);
      setProgress(pct);
      if (elapsed >= totalMs) clearInterval(intervalRef.current);
    }, tickMs);

    // Rotate stage messages
    const stageMs = totalMs / ANALYSIS_STAGES.length;
    let si = 0;
    stageIntervalRef.current = setInterval(() => {
      si++;
      if (si < ANALYSIS_STAGES.length) {
        setStageIndex(si);
      } else {
        clearInterval(stageIntervalRef.current);
      }
    }, stageMs);

    // Fire actual API call simultaneously; wait at least 45s before showing result
    const startTime = Date.now();
    try {
      const response = await api.post('/analyze/link', { url });
      const elapsed2 = Date.now() - startTime;
      const remaining = Math.max(0, totalMs - elapsed2);
      await new Promise((resolve) => setTimeout(resolve, remaining));

      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      setProgress(100);
      await new Promise((res) => setTimeout(res, 400));
      setResult(response.data);
    } catch (err: any) {
      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      setError(err?.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const reset = () => {
    setUrl('');
    setResult(null);
    setProgress(0);
    setStageIndex(0);
    setError('');
  };

  const PlatformIcon = detectPlatformIcon(url);

  return (
    <div className="max-w-4xl mx-auto space-y-10 pb-20">
      {/* Header */}
      <header>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg shadow-violet-200">
            <Link2 size={20} className="text-white" />
          </div>
          <span className="text-xs font-bold text-violet-600 uppercase tracking-widest bg-violet-50 px-3 py-1 rounded-full">
            New Module
          </span>
        </div>
        <h1 className="text-3xl font-heading font-bold text-gray-900">
          Analyze Shared Link
        </h1>
        <p className="text-gray-500 mt-1">
          Paste a YouTube or Instagram link to perform deep forensic analysis on shared social media content.
        </p>
      </header>

      {/* Input Panel */}
      {!analyzing && !result && (
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-[2rem] border border-gray-100 shadow-sm p-10"
        >
          <div className="flex gap-3 mb-3">
            <div className="w-8 h-8 bg-red-50 rounded-xl flex items-center justify-center">
              <Youtube size={16} className="text-red-500" />
            </div>
            <div className="w-8 h-8 bg-pink-50 rounded-xl flex items-center justify-center">
              <Instagram size={16} className="text-pink-500" />
            </div>
          </div>
          <h2 className="text-xl font-bold font-heading mb-6 text-gray-900">
            Paste link for analysis
          </h2>

          <div className="relative mb-6">
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400">
              <PlatformIcon size={20} />
            </div>
            <input
              id="link-input"
              type="url"
              placeholder="https://youtube.com/watch?v=... or https://instagram.com/reel/..."
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setError('');
              }}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
              className="w-full pl-12 pr-4 py-4 bg-gray-50 border border-gray-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-violet-400 focus:border-violet-400 transition-all text-gray-900 font-medium placeholder:text-gray-400"
            />
          </div>

          {error && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-red-500 text-sm font-semibold mb-4 flex items-center gap-2"
            >
              <AlertTriangle size={14} />
              {error}
            </motion.p>
          )}

          <button
            id="analyze-link-btn"
            onClick={handleAnalyze}
            disabled={!url.trim()}
            className="w-full py-4 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-2xl font-bold text-lg  shadow-lg shadow-violet-200 hover:shadow-violet-300 hover:opacity-90 transition-all active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Search size={20} />
            Analyze Link
          </button>

          <p className="text-center text-xs text-gray-400 mt-4 font-medium">
            Powered by multimodal forensic AI · Deep neural authenticity verification
          </p>
        </motion.div>
      )}

      {/* Analysis Animation */}
      <AnimatePresence>
        {analyzing && (
          <AnalysisLogs stages={ANALYSIS_STAGES} progress={progress} stageIndex={stageIndex} fileName={url} />
        )}
      </AnimatePresence>

      {/* Result Panel */}
      <AnimatePresence>
        {result && !analyzing && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Verdict Banner */}
            <div className={`rounded-[2rem] p-8 relative overflow-hidden ${
              result.score === 0
                ? 'bg-gradient-to-br from-emerald-600 to-green-700'
                : result.score >= 80
                ? 'bg-gradient-to-br from-red-900 to-red-950'
                : result.score >= 50
                ? 'bg-gradient-to-br from-orange-900 to-red-900'
                : 'bg-gradient-to-br from-green-900 to-emerald-950'
            }`}>
              <div className="absolute top-0 right-0 w-64 h-64 opacity-10 flex items-center justify-center" style={{ marginTop: '-20px', marginRight: '-20px' }}>
                {result.score >= 50 ? <ShieldX size={200} /> : result.score === 0 ? <CheckCircle2 size={200} /> : <ShieldCheck size={200} />}
              </div>
              <div className="relative z-10">
                <span className={`text-xs font-bold uppercase tracking-widest mb-2 block ${result.score === 0 ? 'text-green-100' : 'text-red-300'}`}>
                  Authenticity Verdict
                </span>
                <h2 className="text-4xl font-heading font-black text-white mb-2">
                  {result.verdict}
                </h2>
                <div className="flex items-center gap-4 mt-4">
                  <div>
                    <p className="text-xs uppercase tracking-widest text-gray-400 mb-1">Platform</p>
                    <p className="text-white font-bold">{result.platform}</p>
                  </div>
                  <div className="w-px h-10 bg-white/10" />
                  <div>
                    <p className="text-xs uppercase tracking-widest text-gray-400 mb-1">Fake Probability Score</p>
                    <p className="text-4xl font-heading font-black text-white">{result.score}%</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Score Meter */}
            <div className="bg-white rounded-[2rem] border border-gray-100 shadow-sm p-8">
              <h3 className="text-lg font-bold font-heading text-gray-900 mb-6 flex items-center gap-2">
                <Activity size={20} className="text-violet-600" />
                Authenticity Score Meter
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm font-bold">
                  <span className="text-green-600">AUTHENTIC</span>
                  <span className="text-red-600">FAKE</span>
                </div>
                <div className="w-full h-5 rounded-full overflow-hidden bg-gradient-to-r from-green-200 via-yellow-200 to-red-200">
                  <div className="relative h-full">
                    <motion.div
                      initial={{ left: 0 }}
                      animate={{ left: `calc(${result.score}% - 14px)` }}
                      transition={{ type: 'spring', stiffness: 60 }}
                      className="absolute top-1/2 -translate-y-1/2 w-7 h-7 bg-gray-900 rounded-full border-4 border-white shadow-lg"
                    />
                  </div>
                </div>
                <div className="flex justify-between text-xs text-gray-400 font-medium">
                  <span>0%</span>
                  <span className="font-bold text-gray-600 text-sm">{result.score}% FAKE</span>
                  <span>100%</span>
                </div>
              </div>
            </div>

            {/* Forensic Findings */}
            <div className="bg-white rounded-[2rem] border border-gray-100 shadow-sm p-8">
              <h3 className="text-lg font-bold font-heading text-gray-900 mb-6 flex items-center gap-2">
                <FileSearch size={20} className="text-violet-600" />
                Forensic Findings
              </h3>
              <div className="space-y-3">
                {result.analysis.map((finding: string, i: number) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -15 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className={`flex items-start gap-3 rounded-2xl p-4 border ${
                        result.score === 0 ? 'bg-green-50 border-green-100' : 'bg-red-50 border-red-100'
                    }`}
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 ${
                        result.score === 0 ? 'bg-green-100' : 'bg-red-100'
                    }`}>
                      <ChevronRight size={12} className={result.score === 0 ? 'text-green-600' : 'text-red-600'} />
                    </div>
                    <p className="text-gray-800 text-sm font-medium leading-relaxed">{finding}</p>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* URL Info */}
            <div className="bg-gray-50 rounded-2xl p-6 border border-gray-100">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">Analyzed URL</p>
              <p className="text-gray-700 font-mono text-sm break-all">{result.url}</p>
            </div>

            {/* Actions */}
            <div className="flex gap-4">
              <button
                onClick={reset}
                className="flex-1 flex items-center justify-center gap-2 py-4 bg-white border border-gray-200 rounded-2xl font-bold text-gray-700 hover:bg-gray-50 transition-all"
              >
                <RotateCcw size={18} />
                Analyze Another Link
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LinkAnalysis;
