import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  X, 
  CheckCircle2, 
  AlertTriangle, 
  FileText, 
  ShieldCheck, 
  Activity, 
  Search,
  Zap,
  Loader2,
  ChevronRight
} from 'lucide-react';
import { analyzeImage } from '../services/analysisService';
import AnalysisLogs, { Stage } from '../components/AnalysisLogs';

const IMAGE_STAGES: Stage[] = [
  { agent: 'System Core', message: 'Initializing async memory buffer & PIL conversion...' },
  { agent: 'Forensic Engine', message: 'Generating Error Level Analysis (ELA) map from raw pixels...' },
  { agent: 'Vision AI (M1)', message: 'Running EfficientNet-B0 global AI synthesis detector...' },
  { agent: 'Manipulation Scanner (M2)', message: 'Analyzing 6-channel stacked ResNet tensors for splicing anomalies...' },
  { agent: 'Source Identifier (M3)', message: 'Cross-referencing generative signatures with SDXL dataset...' },
  { agent: 'Anomaly Localizer (M4)', message: 'Scanning regional bounding boxes for manipulation anomalies...' },
  { agent: 'Compression Analyzer (M5)', message: 'Evaluating upscaling latency and block compression artifacts...' },
  { agent: 'Consensus Engine', message: 'Computing exact deterministic multi-model verdict...' },
  { agent: 'Report Generator', message: 'Generating dynamic natural language explanation...' },
  { agent: 'System Engine', message: 'Creating Base64 ELA visualization overlay for UI...' },
  { agent: 'System Core', message: 'Finalizing response payload.' }
];

const ImageAnalysis = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [progress, setProgress] = useState(0);
  const [stageIndex, setStageIndex] = useState(0);
  const intervalRef = React.useRef<any>(null);
  const stageIntervalRef = React.useRef<any>(null);

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
    setLoading(true);
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
    
    const stageMs = totalMs / IMAGE_STAGES.length;
    let si = 0;
    stageIntervalRef.current = setInterval(() => {
      si++;
      if (si < IMAGE_STAGES.length) {
        setStageIndex(si);
      } else {
        clearInterval(stageIntervalRef.current);
      }
    }, stageMs);

    const startTime = Date.now();
    try {
      const data = await analyzeImage(file);
      const remaining = Math.max(0, totalMs - (Date.now() - startTime));
      await new Promise(r => setTimeout(r, remaining));
      
      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      setProgress(100);
      setStageIndex(IMAGE_STAGES.length - 1);
      setTimeout(() => setResult(data), 500);
    } catch (err) {
      console.error(err);
      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      alert("Analysis failed.");
    } finally {
      setTimeout(() => setLoading(false), 500);
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
        <h1 className="text-3xl font-heading font-bold text-gray-900">AI Image Analysis</h1>
        <p className="text-gray-500 mt-1">Upload an image to detect AI generation or manual manipulations.</p>
      </header>

      {!loading && !result && (
        <motion.div 
            layout
            className="bg-white p-12 rounded-[2rem] border-2 border-dashed border-gray-200 flex flex-col items-center justify-center transition-colors hover:border-primary-300"
        >
          {!file ? (
            <div className="text-center">
              <div className="w-24 h-24 bg-primary-50 text-primary-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Upload size={40} />
              </div>
              <h3 className="text-2xl font-bold font-heading mb-4">Drop your image here</h3>
              <p className="text-gray-400 mb-8 max-w-sm mx-auto">Supports JPG, PNG and JPEG. Max file size: 10MB.</p>
              <label className="btn-primary cursor-pointer inline-flex items-center gap-2">
                <input type="file" className="hidden" accept="image/*" onChange={onFileChange} />
                Browse Files
              </label>
            </div>
          ) : (
            <div className="w-full max-w-2xl">
              <div className="relative rounded-2xl overflow-hidden shadow-2xl mb-8 bg-gray-50 aspect-video flex items-center justify-center">
                <img src={preview!} alt="Preview" className="max-h-full object-contain" />
                <button 
                    onClick={reset}
                    className="absolute top-4 right-4 p-2 bg-white/80 backdrop-blur rounded-full text-red-500 hover:bg-white transition-colors"
                >
                    <X size={20} />
                </button>
              </div>
              
              <div className="flex flex-col gap-6">
                <div className="flex justify-between items-center bg-gray-50 p-4 rounded-xl">
                   <div className="flex items-center gap-3">
                       <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                           <FileText size={20} className="text-gray-400" />
                       </div>
                       <div>
                           <p className="text-sm font-bold text-gray-900 truncate max-w-[200px]">{file.name}</p>
                           <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                       </div>
                   </div>
                </div>

                <button onClick={handleUpload} className="btn-primary w-full py-4 text-lg">
                    Start Analysis
                </button>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {loading && (
        <AnalysisLogs stages={IMAGE_STAGES} progress={progress} stageIndex={stageIndex} fileName={file?.name} />
      )}

      {result && !loading && (
        <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid lg:grid-cols-3 gap-8"
        >
          {/* Main Result Card */}
          <div className="lg:col-span-2 space-y-8">
            <div className="bg-white p-8 rounded-[2rem] border border-gray-100 shadow-sm">
                <div className="flex justify-between items-start mb-10">
                    <div>
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">Final Verdict</span>
                        <h2 className={`text-4xl font-heading font-bold ${result.confidence > 0.5 && result.verdict !== 'AUTHENTIC' ? 'text-red-500' : 'text-green-500'}`}>
                            {result.verdict.replace(/_/g, ' ')}
                        </h2>
                    </div>
                    <div className="text-right">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">Confidence</span>
                        <p className="text-4xl font-heading font-bold text-gray-900">{Math.round(result.confidence * 100)}%</p>
                    </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-10">
                    {[
                        { label: 'AI Detection', score: Math.round(result.scores.ai_detector * 100) + '%', icon: Zap, status: result.scores.ai_detector > 0.5 ? 'Suspicious' : 'Clean' },
                        { label: 'Manipulation', score: Math.round(result.scores.manipulation * 100) + '%', icon: Search, status: result.scores.manipulation > 0.5 ? 'Suspicious' : 'Clean' },
                        { label: 'Attribution', score: result.generator || 'N/A', icon: ShieldCheck, status: result.scores.source_id > 0.5 ? 'Found' : 'Clear' },
                        { label: 'Anomalies', score: Math.round(result.scores.patch_anomaly * 100) + '%', icon: AlertTriangle, status: result.scores.patch_anomaly > 0.5 ? 'Detected' : 'Clean' },
                        { label: 'Upscaled', score: Math.round(result.scores.compression * 100) + '%', icon: Activity, status: result.scores.compression > 0.5 ? 'Yes' : 'No' },
                    ].map((item, i) => (
                        <div key={i} className="bg-gray-50 p-4 rounded-2xl border border-gray-100 flex flex-col items-center text-center">
                            <item.icon size={20} className="text-primary-600 mb-3" />
                            <p className="text-[10px] font-bold text-gray-400 uppercase mb-1">{item.label}</p>
                            <p className="text-lg font-bold text-gray-900 truncate w-full">{item.score}</p>
                            <span className={`text-[10px] font-bold px-2 py-0.5 mt-2 rounded-full ${
                                ['Clean', 'Clear', 'No'].includes(item.status)
                                ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                            }`}>{item.status}</span>
                        </div>
                    ))}
                </div>

                <div className="bg-primary-50 rounded-2xl p-6 border border-primary-100">
                    <div className="flex items-center gap-3 mb-4">
                        <ShieldCheck className="text-primary-600" size={24} />
                        <h3 className="font-bold text-primary-900">Analysis Explanation</h3>
                    </div>
                    <p className="text-primary-800 leading-relaxed font-medium">
                        {result.explanation}
                    </p>
                </div>

                <div className="mt-8">
                    <h3 className="text-xl font-bold font-heading mb-4 text-gray-900 border-b border-gray-100 pb-3">Deep-Dive Assessment Layers</h3>
                    <div className="space-y-3">
                        <div className="flex items-center justify-between p-4 bg-gray-50 hover:bg-white transition-colors rounded-xl border border-gray-100 shadow-sm">
                            <span className="font-bold text-gray-700 flex items-center gap-2"><Zap size={16} className="text-primary-500" /> Layer 1: Generative Pattern AI</span>
                            <span className="font-mono text-gray-900 font-bold bg-white px-3 py-1 rounded-lg border border-gray-100">{Math.round(result.scores.ai_detector * 100)}%</span>
                        </div>
                        <div className="flex items-center justify-between p-4 bg-gray-50 hover:bg-white transition-colors rounded-xl border border-gray-100 shadow-sm">
                            <span className="font-bold text-gray-700 flex items-center gap-2"><Search size={16} className="text-primary-500" /> Layer 2: Pixel Manipulation Scanner</span>
                            <span className="font-mono text-gray-900 font-bold bg-white px-3 py-1 rounded-lg border border-gray-100">{Math.round(result.scores.manipulation * 100)}%</span>
                        </div>
                        <div className="flex items-center justify-between p-4 bg-gray-50 hover:bg-white transition-colors rounded-xl border border-gray-100 shadow-sm">
                            <span className="font-bold text-gray-700 flex items-center gap-2"><ShieldCheck size={16} className="text-primary-500" /> Layer 3: Generator Source Identifier</span>
                            <span className="font-mono text-gray-900 font-bold bg-white px-3 py-1 rounded-lg border border-gray-100">{result.generator || 'No distinct signature'}</span>
                        </div>
                        <div className="flex items-center justify-between p-4 bg-gray-50 hover:bg-white transition-colors rounded-xl border border-gray-100 shadow-sm">
                            <span className="font-bold text-gray-700 flex items-center gap-2"><AlertTriangle size={16} className="text-primary-500" /> Layer 4: Regional Patch Anomaly</span>
                            <span className="font-mono text-gray-900 font-bold bg-white px-3 py-1 rounded-lg border border-gray-100">{Math.round(result.scores.patch_anomaly * 100)}%</span>
                        </div>
                        <div className="flex items-center justify-between p-4 bg-gray-50 hover:bg-white transition-colors rounded-xl border border-gray-100 shadow-sm">
                            <span className="font-bold text-gray-700 flex items-center gap-2"><Activity size={16} className="text-primary-500" /> Layer 5: Compression & Upscaling</span>
                            <span className="font-mono text-gray-900 font-bold bg-white px-3 py-1 rounded-lg border border-gray-100">{Math.round(result.scores.compression * 100)}%</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Structured Report / ELA Map */}
            <div className="bg-white p-8 rounded-[2rem] border border-gray-100 shadow-sm relative overflow-hidden">
                <div className="absolute top-0 right-0 p-8 opacity-5">
                    <FileText size={120} />
                </div>
                <h3 className="text-2xl font-bold font-heading mb-8 relative z-10">ELA Forensic Map</h3>
                <div className="space-y-6 relative z-10">
                    {result.ela_map_base64 ? (
                        <div className="rounded-xl overflow-hidden border border-gray-200">
                           <img 
                               src={`data:image/png;base64,${result.ela_map_base64}`} 
                               alt="ELA overlay" 
                               className="w-full object-cover"
                           />
                        </div>
                    ) : (
                        <p className="text-gray-500 italic">No ELA map generated.</p>
                    )}
                    <p className="text-sm text-gray-600 pt-2">Brighter error blocks in the ELA map indicate higher levels of compression inconsistency, frequently revealing traces of digital modification or splicing.</p>
                </div>
               
                <div className="mt-10 pt-8 border-t border-gray-50 flex justify-center relative z-10">
                    <button onClick={reset} className="btn-primary w-full md:w-auto px-12">Run New Analysis</button>
                </div>
            </div>
          </div>

          {/* Sider Preview */}
          <div className="space-y-8">
            <div className="bg-gray-900 p-4 rounded-3xl shadow-xl border-4 border-white aspect-[3/4] overflow-hidden sticky top-24">
                <img src={preview!} alt="Final" className="w-full h-full object-cover rounded-2xl overflow-hidden opacity-90" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent flex flex-col justify-end p-8">
                    <div className="flex gap-2 mb-4">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                        <span className="text-[10px] font-bold text-white uppercase tracking-widest">AI Heatmap Active</span>
                    </div>
                    <p className="text-white font-bold text-lg mb-1">{file?.name}</p>
                    <p className="text-gray-400 text-xs">Analyzed on {new Date().toLocaleDateString()}</p>
                </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ImageAnalysis;
