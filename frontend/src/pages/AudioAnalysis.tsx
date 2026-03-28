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
  { agent: 'Audio Gateway', message: 'Initializing 6-phase acoustic forensic pipeline...' },
  { agent: 'Phase 1 (Meta)', message: 'Extracting codec signatures, bitrate mapping, and header anomalies...' },
  { agent: 'Signal Process', message: 'Demultiplexing waveform arrays and sampling Nyquist thresholds...' },
  { agent: 'Phase 2 (Spec)', message: 'Computing Mel-spectrograms for high-frequency hard cutoffs...' },
  { agent: 'Cepstral AI', message: 'Extracting MFCCs to detect unnatural vocoder regularity...' },
  { agent: 'Phase 3 (Temp)', message: 'Isolating bio-signals: measuring sub-vocal F0 pitch jitter...' },
  { agent: 'Respiration AI', message: 'Scanning temporal voids for authentic human inhalation/exhalation events...' },
  { agent: 'Phase 4 (Neural)', message: 'Booting Nexus-AudioForge-v2 transformer (wav2vec2-base-960h)...' },
  { agent: 'Phase 5 (Spkr)', message: 'Computing cosine similarity across sequential speaker embeddings...' },
  { agent: 'Noise Model', message: 'Validating stationary background noise floor consistency...' },
  { agent: 'Phase 6 (Fusion)', message: 'Reconciling 25% neural weights with 75% biometric signal flags...' },
  { agent: 'Audio Gateway', message: 'Forensic telemetry packaged and finalized.' }
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
            className="space-y-8"
        >
          {/* Main Verdict Card */}
          <div className="bg-white p-8 rounded-[2rem] border border-gray-100 shadow-sm">
            <div className="flex justify-between items-start mb-8">
              <div>
                <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">Final Verdict</span>
                <h2 className={`text-4xl font-heading font-bold ${result.score > 50 ? 'text-red-500' : 'text-green-500'}`}>
                  {result.verdict}
                </h2>
                {result.details?.explanation && (
                  <p className="text-sm text-gray-500 mt-2 max-w-lg leading-relaxed">{result.details.explanation}</p>
                )}
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">Fake Score</span>
                <p className={`text-5xl font-heading font-bold ${result.score > 50 ? 'text-red-500' : 'text-green-500'}`}>{result.score}%</p>
                {result.details?.composite && (
                  <p className="text-xs text-gray-400 mt-1">
                    Neural {Math.round(result.details.composite.neural_score * 100)}% · Signal {Math.round(result.details.composite.signal_score * 100)}%
                  </p>
                )}
              </div>
            </div>

            {/* Forensic Flags */}
            {result.details?.flags && result.details.flags.length > 0 && (
              <div className="bg-amber-50 rounded-2xl p-5 border border-amber-100 mb-6">
                <div className="flex items-center gap-2 mb-3">
                  <AlertTriangle className="text-amber-600" size={20} />
                  <h3 className="font-bold text-amber-900 text-sm">Forensic Anomalies ({result.details.flags.length})</h3>
                </div>
                <ul className="list-disc pl-5 space-y-1 text-amber-800 text-sm">
                  {result.details.flags.map((flag: string, i: number) => (
                    <li key={i}>{flag}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Module Scores Grid */}
          {result.details?.module_scores && (
            <div className="bg-white p-8 rounded-[2rem] border border-gray-100 shadow-sm">
              <h3 className="font-bold text-gray-900 mb-6 flex items-center gap-2">
                <Zap size={18} className="text-teal-600" /> Module Forensic Scores
                <span className="text-xs text-gray-400 font-normal ml-1">(0 = authentic, 1 = suspicious)</span>
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(result.details.module_scores).map(([mod, val]: [string, any]) => {
                  const pct = Math.round(val * 100);
                  const color = pct > 60 ? 'bg-red-500' : pct > 40 ? 'bg-amber-400' : 'bg-green-500';
                  return (
                    <div key={mod} className="bg-gray-50 rounded-2xl p-4 border border-gray-100">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-bold text-gray-600 capitalize">{mod}</span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-lg text-white ${color}`}>{pct}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div className={`h-1.5 rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Temporal + Speaker + Spectral Details */}
          <div className="grid md:grid-cols-3 gap-6">
            {result.details?.temporal && Object.keys(result.details.temporal).length > 0 && (
              <div className="bg-white p-6 rounded-[2rem] border border-gray-100 shadow-sm">
                <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2 text-sm">
                  <Activity size={16} className="text-teal-600" /> Temporal Bio-Signals
                </h3>
                <div className="space-y-3">
                  {result.details.temporal.breath_rate_per_minute !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Breath Rate</span>
                      <span className="font-bold text-gray-800">{result.details.temporal.breath_rate_per_minute} /min</span>
                    </div>
                  )}
                  {result.details.temporal.f0_jitter !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">F0 Jitter</span>
                      <span className="font-bold text-gray-800">{result.details.temporal.f0_jitter}</span>
                    </div>
                  )}
                  {result.details.temporal.f0_range_hz !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Pitch Range</span>
                      <span className="font-bold text-gray-800">{result.details.temporal.f0_range_hz} Hz</span>
                    </div>
                  )}
                  {result.details.temporal.voiced_ratio !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Voiced Ratio</span>
                      <span className="font-bold text-gray-800">{Math.round(result.details.temporal.voiced_ratio * 100)}%</span>
                    </div>
                  )}
                  {result.details.temporal.pause_count !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Natural Pauses</span>
                      <span className="font-bold text-gray-800">{result.details.temporal.pause_count}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {result.details?.speaker && Object.keys(result.details.speaker).length > 0 && (
              <div className="bg-white p-6 rounded-[2rem] border border-gray-100 shadow-sm">
                <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2 text-sm">
                  <ShieldCheck size={16} className="text-teal-600" /> Speaker Consistency
                </h3>
                <div className="space-y-3">
                  {result.details.speaker.segment_count !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Segments</span>
                      <span className="font-bold text-gray-800">{result.details.speaker.segment_count}</span>
                    </div>
                  )}
                  {result.details.speaker.mean_segment_similarity !== undefined && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Voice Similarity</span>
                      <span className="font-bold text-gray-800">{Math.round(result.details.speaker.mean_segment_similarity * 100)}%</span>
                    </div>
                  )}
                  {result.details.speaker.note && (
                    <p className="text-xs text-gray-400 mt-2 leading-relaxed">{result.details.speaker.note}</p>
                  )}
                </div>
              </div>
            )}

            {result.details?.neural_model && (
              <div className="bg-white p-6 rounded-[2rem] border border-gray-100 shadow-sm">
                <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2 text-sm">
                  <Search size={16} className="text-teal-600" /> Neural Model
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Model</span>
                    <span className="font-bold text-gray-800 text-right text-xs">{result.details.neural_model.name}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Label</span>
                    <span className={`font-bold ${result.details.neural_model.label === 'SYNTHETIC' ? 'text-red-500' : result.details.neural_model.label === 'AUTHENTIC' ? 'text-green-500' : 'text-gray-500'}`}>
                      {result.details.neural_model.label}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Confidence</span>
                    <span className="font-bold text-gray-800">{Math.round(result.details.neural_model.confidence * 100)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Score</span>
                    <span className="font-bold text-gray-800">{Math.round(result.details.neural_model.score * 100)}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-start">
            <button onClick={reset} className="py-3 px-8 rounded-xl font-bold border border-gray-200 text-gray-700 bg-white hover:bg-gray-50 transition-colors">
              ← New Audio Scan
            </button>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default AudioAnalysis;
