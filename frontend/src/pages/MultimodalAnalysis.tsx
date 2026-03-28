import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle2, 
  AlertTriangle, 
  Activity, 
  X,
  Upload,
  Image as ImageIcon,
  Video as VideoIcon,
  Mic,
  FileText,
  ShieldAlert
} from 'lucide-react';
import { analyzeMultimodal } from '../services/analysisService';
import AnalysisLogs, { Stage } from '../components/AnalysisLogs';

const MULTIMODAL_STAGES: Stage[] = [
  { agent: 'Multimodal Core', message: 'Initializing cross-modal analysis engine...' },
  { agent: 'Input Parser', message: 'Detecting input types...' },
  { agent: 'Image Engine', message: 'Running image forensic analysis...' },
  { agent: 'Audio Engine', message: 'Processing voice authenticity...' },
  { agent: 'Text Analyzer', message: 'Evaluating semantic consistency...' },
  { agent: 'Cross-Modal Engine', message: 'Comparing modalities...' },
  { agent: 'Consistency Checker', message: 'Detecting logical mismatches...' },
  { agent: 'Fact Checker', message: 'Validating combined content integrity...' },
  { agent: 'Anomaly Detector', message: 'Identifying cross-modal inconsistencies...' },
  { agent: 'Decision Engine', message: 'Computing final authenticity score...' },
];

const MultimodalAnalysis = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [text, setText] = useState('');
  
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  
  const intervalRef = useRef<any>(null);
  const stageIntervalRef = useRef<any>(null);

  const handleUpload = async () => {
    if (!imageFile && !videoFile && !audioFile && !text) {
        alert("Please provide at least one input modality.");
        return;
    }
    setAnalyzing(true);
    setProgress(0);
    setStageIndex(0);
    
    // Total animation time 45-60s
    const totalMs = 50000;
    const tickMs = 200;
    let elapsed = 0;
    
    intervalRef.current = setInterval(() => {
      elapsed += tickMs;
      const pct = Math.min((elapsed / totalMs) * 100, 99);
      setProgress(pct);
      if (elapsed >= totalMs) clearInterval(intervalRef.current);
    }, tickMs);
    
    const stageMs = totalMs / MULTIMODAL_STAGES.length;
    let si = 0;
    stageIntervalRef.current = setInterval(() => {
      si++;
      if (si < MULTIMODAL_STAGES.length) {
        setStageIndex(si);
      } else {
        clearInterval(stageIntervalRef.current);
      }
    }, stageMs);

    const startTime = Date.now();
    try {
      const data = await analyzeMultimodal({
          image: imageFile,
          video: videoFile,
          audio: audioFile,
          text: text
      });
      const remaining = Math.max(0, totalMs - (Date.now() - startTime));
      await new Promise(r => setTimeout(r, remaining));
      
      clearInterval(intervalRef.current);
      clearInterval(stageIntervalRef.current);
      setProgress(100);
      setStageIndex(MULTIMODAL_STAGES.length - 1);
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
    setImageFile(null);
    setVideoFile(null);
    setAudioFile(null);
    setText('');
    setResult(null);
    setProgress(0);
    setStageIndex(0);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10 pb-20">
      <header>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-indigo-500 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-200">
            <Activity size={20} className="text-white" />
          </div>
        </div>
        <h1 className="text-3xl font-heading font-bold text-gray-900">Multimodal Analysis</h1>
        <p className="text-gray-500 mt-1">Simultaneous cross-modal verification across Image, Video, Audio and Text.</p>
      </header>

      {!analyzing && !result && (
        <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white p-10 rounded-[2rem] border border-gray-100 shadow-sm"
        >
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            {/* Image */}
            <div className="border border-dashed border-gray-300 rounded-2xl p-6 flex flex-col items-center justify-center relative hover:border-indigo-400 transition-colors">
                {imageFile && (
                    <button onClick={() => setImageFile(null)} className="absolute top-2 right-2 p-1 bg-gray-100 rounded-full hover:bg-red-100 text-red-500">
                        <X size={16} />
                    </button>
                )}
                <ImageIcon size={32} className={imageFile ? "text-indigo-600 mb-3" : "text-gray-300 mb-3"} />
                <p className="font-bold text-sm text-gray-700">{imageFile ? imageFile.name : 'Upload Image'}</p>
                {!imageFile && (
                    <label className="mt-4 px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg text-xs font-bold cursor-pointer hover:bg-indigo-100">
                        <input type="file" className="hidden" accept="image/*" onChange={(e) => e.target.files && setImageFile(e.target.files[0])} />
                        Choose File
                    </label>
                )}
            </div>
            
            {/* Video */}
            <div className="border border-dashed border-gray-300 rounded-2xl p-6 flex flex-col items-center justify-center relative hover:border-indigo-400 transition-colors">
                {videoFile && (
                    <button onClick={() => setVideoFile(null)} className="absolute top-2 right-2 p-1 bg-gray-100 rounded-full hover:bg-red-100 text-red-500">
                        <X size={16} />
                    </button>
                )}
                <VideoIcon size={32} className={videoFile ? "text-indigo-600 mb-3" : "text-gray-300 mb-3"} />
                <p className="font-bold text-sm text-gray-700">{videoFile ? videoFile.name : 'Upload Video'}</p>
                {!videoFile && (
                    <label className="mt-4 px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg text-xs font-bold cursor-pointer hover:bg-indigo-100">
                        <input type="file" className="hidden" accept="video/*" onChange={(e) => e.target.files && setVideoFile(e.target.files[0])} />
                        Choose File
                    </label>
                )}
            </div>

            {/* Audio */}
            <div className="border border-dashed border-gray-300 rounded-2xl p-6 flex flex-col items-center justify-center relative hover:border-indigo-400 transition-colors">
                {audioFile && (
                    <button onClick={() => setAudioFile(null)} className="absolute top-2 right-2 p-1 bg-gray-100 rounded-full hover:bg-red-100 text-red-500">
                        <X size={16} />
                    </button>
                )}
                <Mic size={32} className={audioFile ? "text-indigo-600 mb-3" : "text-gray-300 mb-3"} />
                <p className="font-bold text-sm text-gray-700">{audioFile ? audioFile.name : 'Upload Audio'}</p>
                {!audioFile && (
                    <label className="mt-4 px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg text-xs font-bold cursor-pointer hover:bg-indigo-100">
                        <input type="file" className="hidden" accept="audio/*" onChange={(e) => e.target.files && setAudioFile(e.target.files[0])} />
                        Choose File
                    </label>
                )}
            </div>

            {/* Text Context */}
            <div className="border border-gray-200 rounded-2xl p-6 flex flex-col">
                <div className="flex items-center gap-2 mb-3">
                    <FileText size={20} className="text-gray-400" />
                    <span className="font-bold text-gray-700 text-sm">Contextual Text</span>
                </div>
                <textarea 
                    className="w-full flex-1 bg-gray-50 border-none rounded-xl p-4 text-sm focus:ring-2 focus:ring-indigo-400 resize-none"
                    placeholder="Enter social media caption, news headline or contextual metadata..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                />
            </div>
          </div>

          <button 
              onClick={handleUpload} 
              disabled={!imageFile && !videoFile && !audioFile && !text}
              className="w-full py-4 text-white bg-indigo-600 rounded-2xl font-bold text-lg hover:bg-indigo-700 shadow-xl shadow-indigo-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
              Analyze Combinations
          </button>
        </motion.div>
      )}

      {analyzing && (
        <AnalysisLogs stages={MULTIMODAL_STAGES} progress={progress} stageIndex={stageIndex} fileName="Multimodal Payload" />
      )}

      {result && !analyzing && (
        <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-6"
        >
            <div className={`p-10 rounded-[2rem] text-white border-2 relative overflow-hidden ${result.score > 50 ? 'bg-zinc-900 border-red-500/30' : 'bg-zinc-900 border-green-500/30'}`}>
                <div className="absolute top-0 right-0 w-64 h-64 bg-red-500/10 blur-3xl rounded-full" />
                
                <h2 className="text-sm font-bold tracking-widest uppercase mb-4 text-gray-400">Final Verdict</h2>
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                    <div>
                        <h1 className={`text-5xl font-black font-heading tracking-tight mb-2 ${result.score > 50 ? 'text-red-500' : 'text-green-400'}`}>
                            {result.verdict}
                        </h1>
                        <p className="text-xl text-gray-300 font-medium">Score: {result.score}%</p>
                    </div>
                    
                    <button onClick={reset} className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white rounded-xl font-bold transition-colors">
                        New Analysis
                    </button>
                </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-white p-8 rounded-[2rem] border border-gray-100 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <ShieldAlert className="text-indigo-600" size={24} />
                        <h3 className="font-bold text-lg text-gray-900">Reasoning</h3>
                    </div>
                    {result.details.flags && result.details.flags.length > 0 ? (
                        <ul className="space-y-4">
                            {result.details.flags.map((flag: string, i: number) => (
                                <li key={i} className="flex items-start gap-3 text-gray-700">
                                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 mt-2 block flex-shrink-0" />
                                    <span>{flag}</span>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="text-gray-600">No cross-modal inconsistencies detected.</p>
                    )}
                </div>

                <div className="bg-indigo-50 p-8 rounded-[2rem] border border-indigo-100">
                    <h3 className="font-bold text-lg text-indigo-900 mb-4 uppercase tracking-widest text-xs">FACT CHECK RESULT:</h3>
                    <p className="text-indigo-800 text-lg leading-relaxed font-medium italic">
                        "{result.score > 50 ? 'This content is likely manipulated and does not represent real-world authenticity.' : 'This multi-media content appears consistent with natural authentic capture.'}"
                    </p>
                    <div className="mt-8 flex gap-2">
                        {['Image', 'Video', 'Audio', 'Text'].map(mod => (
                            <span key={mod} className={`px-3 py-1 rounded-full text-xs font-bold ${
                                (mod === 'Image' && result.details.modalities >= 1) || 
                                (mod === 'Video' && videoFile) || 
                                (mod === 'Audio' && audioFile) || 
                                (mod === 'Text' && text) ? 'bg-indigo-200 text-indigo-800' : 'bg-indigo-100/50 text-indigo-300'
                            }`}>{mod}</span>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
      )}
    </div>
  );
};

export default MultimodalAnalysis;
