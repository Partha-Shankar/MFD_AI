import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Mic, 
  X, 
  CheckCircle2, 
  AlertTriangle, 
  Activity, 
  Search,
  Zap,
  ShieldCheck
} from 'lucide-react';
import { analyzeAudio } from '../services/analysisService';
import AnalysisLogs, { Stage } from '../components/AnalysisLogs';

const AUDIO_STAGES: Stage[] = [
  { agent: 'Audio Core', message: 'Initializing audio forensic engine...' },
  { agent: 'Signal Processor', message: 'Loading waveform data...' },
  { agent: 'Voice Analyzer', message: 'Extracting speech features...' },
  { agent: 'Neural Detector', message: 'Scanning for synthetic voice patterns...' },
  { agent: 'Frequency Engine', message: 'Performing spectral analysis...' },
  { agent: 'Noise Model', message: 'Evaluating background noise consistency...' },
  { agent: 'Compression Inspector', message: 'Checking encoding artifacts...' },
  { agent: 'Prosody Analyzer', message: 'Evaluating pitch and rhythm patterns...' },
  { agent: 'Biometric Agent', message: 'Analyzing speaker uniqueness...' },
  { agent: 'Cloning Detector', message: 'Detecting voice replication signatures...' },
  { agent: 'Emotion Engine', message: 'Evaluating emotional variability...' },
  { agent: 'Temporal Audio', message: 'Checking waveform continuity...' },
  { agent: 'Consensus Engine', message: 'Aggregating audio forensic signals...' },
  { agent: 'Confidence Model', message: 'Computing authenticity score...' },
  { agent: 'Report Generator', message: 'Generating explainable insights...' },
  { agent: 'Audio Core', message: 'Analysis complete.' }
];

const AudioAnalysis = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  
  const intervalRef = useRef<any>(null);
  const stageIntervalRef = useRef<any>(null);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setAnalyzing(true);
    setProgress(0);
    setStageIndex(0);
    
    // Total animation time 45s
    const totalMs = 45000;
    const tickMs = 200;
    let elapsed = 0;
    
    intervalRef.current = setInterval(() => {
      elapsed += tickMs;
      const pct = Math.min((elapsed / totalMs) * 100, 99);
      setProgress(pct);
      if (elapsed >= totalMs) clearInterval(intervalRef.current);
    }, tickMs);
    
    const stageMs = totalMs / AUDIO_STAGES.length;
    let si = 0;
    stageIntervalRef.current = setInterval(() => {
      si++;
      if (si < AUDIO_STAGES.length) {
        setStageIndex(si);
      } else {
        clearInterval(stageIntervalRef.current);
      }
    }, stageMs);

    const startTime = Date.now();
    try {
      const data = await analyzeAudio(file);
      const remaining = Math.max(0, totalMs - (Date.now() - startTime));
      await new Promise(r => setTimeout(r, remaining));
      
      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      setProgress(100);
      setStageIndex(AUDIO_STAGES.length - 1);
      setTimeout(() => setResult(data), 500);
    } catch (err) {
      console.error(err);
      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      alert("Analysis failed.");
    } finally {
      setTimeout(() => setAnalyzing(false), 500);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setProgress(0);
    setStageIndex(0);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10 pb-20">
      <header>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-teal-500 rounded-2xl flex items-center justify-center shadow-lg shadow-teal-200">
            <Mic size={20} className="text-white" />
          </div>
        </div>
        <h1 className="text-3xl font-heading font-bold text-gray-900">Audio Voice Analysis</h1>
        <p className="text-gray-500 mt-1">Detect cloned, synthesized or manipulated voice patterns.</p>
      </header>

      {!analyzing && !result && (
        <motion.div 
            layout
            className="bg-white p-12 rounded-[2rem] border-2 border-dashed border-gray-200 flex flex-col items-center justify-center transition-colors hover:border-teal-300"
        >
          {!file ? (
            <div className="text-center">
              <div className="w-24 h-24 bg-teal-50 text-teal-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Mic size={40} />
              </div>
              <h3 className="text-2xl font-bold font-heading mb-4">Drop your audio file here</h3>
              <p className="text-gray-400 mb-8 max-w-sm mx-auto">Supports MP3, WAV and OGG. Max file size: 10MB.</p>
              <label className="bg-teal-600 text-white px-8 py-4 rounded-xl font-bold cursor-pointer inline-flex items-center gap-2 hover:bg-teal-700 transition-all active:scale-95 shadow-lg shadow-teal-200">
                <input type="file" className="hidden" accept="audio/*" onChange={onFileChange} />
                Browse Audio Files
              </label>
            </div>
          ) : (
            <div className="w-full max-w-2xl">
              <div className="relative rounded-2xl overflow-hidden shadow-2xl mb-8 bg-black p-8 flex items-center justify-center">
                <audio src={preview!} controls className="w-full" />
                <button 
                    onClick={reset}
                    className="absolute top-4 right-4 p-2 bg-white/20 backdrop-blur rounded-full text-white hover:bg-white/40 transition-colors"
                >
                    <X size={20} />
                </button>
              </div>
              
              <div className="flex flex-col gap-6">
                <div className="flex justify-between items-center bg-gray-50 p-4 rounded-xl">
                   <div className="flex items-center gap-3">
                       <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                           <Mic size={20} className="text-teal-500" />
                       </div>
                       <div>
                           <p className="text-sm font-bold text-gray-900 truncate max-w-[200px]">{file.name}</p>
                           <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                       </div>
                   </div>
                </div>

                <button onClick={handleUpload} className="bg-teal-600 text-white w-full py-4 rounded-xl font-bold text-lg shadow-lg shadow-teal-200">
                    Start Audio Analysis
                </button>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {analyzing && (
        <AnalysisLogs stages={AUDIO_STAGES} progress={progress} stageIndex={stageIndex} fileName={file?.name} />
      )}

      {result && !analyzing && (
        <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid lg:grid-cols-3 gap-8"
        >
          <div className="lg:col-span-2 space-y-8">
            <div className="bg-white p-8 rounded-[2rem] border border-gray-100 shadow-sm relative overflow-hidden">
                <div className="flex justify-between items-start mb-10">
                    <div>
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">Final Verdict</span>
                        <h2 className={`text-4xl font-heading font-bold ${result.score > 50 ? 'text-red-500' : 'text-green-500'}`}>
                            {result.verdict}
                        </h2>
                    </div>
                    <div className="text-right">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">Fake Probability</span>
                        <p className="text-4xl font-heading font-bold text-gray-900">{result.score}%</p>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-10">
                  <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100 flex flex-col items-center justify-center text-center">
                    <Activity size={24} className="text-teal-600 mb-2" />
                    <p className="text-sm font-bold text-gray-900">Spectral Consistency</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-2xl border border-gray-100 flex flex-col items-center justify-center text-center">
                    <ShieldCheck size={24} className="text-teal-600 mb-2" />
                    <p className="text-sm font-bold text-gray-900">Pattern Continuity</p>
                  </div>
                </div>

                <div className="bg-teal-50 rounded-2xl p-6 border border-teal-100">
                    <div className="flex items-center gap-3 mb-4">
                        <AlertTriangle className="text-teal-600" size={24} />
                        <h3 className="font-bold text-teal-900">Forensic Flags</h3>
                    </div>
                    {result.details.flags && result.details.flags.length > 0 ? (
                      <ul className="list-disc pl-5 space-y-2 text-teal-800">
                        {result.details.flags.map((flag: string, i: number) => (
                          <li key={i}>{flag}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-teal-800 font-medium">No critical manipulation flags detected.</p>
                    )}
                </div>
            </div>
            
            <div className="mt-6 flex justify-between">
                <button onClick={reset} className="btn-secondary py-3 px-6 rounded-xl font-bold border border-gray-200 text-gray-700 bg-white">New Audio Scan</button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default AudioAnalysis;
