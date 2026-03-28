import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Video as VideoIcon, 
  X, 
  Play, 
  ShieldCheck, 
  AlertCircle,
  Activity,
  History,
  TrendingUp,
  Loader2
} from 'lucide-react';
import { analyzeVideo } from '../services/analysisService';
import AnalysisLogs, { Stage } from '../components/AnalysisLogs';

const VIDEO_STAGES: Stage[] = [
  { agent: 'System Core', message: 'Initializing multi-agent video forensic pipeline...' },
  { agent: 'Media Parser', message: 'Extracting MP4 headers and decoding visual streams...' },
  { agent: 'Frame Engine', message: 'Sampling dynamic keyframes across batch iterations...' },
  { agent: 'Spatial Detector', message: 'Running dual-layer CNN for frame-level manipulation scanning...' },
  { agent: 'Face Tracker', message: 'Stabilizing landmarks and cropping facial regions of interest...' },
  { agent: 'Temporal Analyzer', message: 'Computing optical flow matrices to detect unnatural motion vectors...' },
  { agent: 'Interpolation Scanner', message: 'Inspecting frame interpolation boundaries for RIFE/DAIN artifacts...' },
  { agent: 'Signal Process', message: 'Demultiplexing embedded AAC/WAV audio track from container...' },
  { agent: 'Cross-Modal Engine', message: 'Synchronizing audio phonemes with visual lip kinematics...' },
  { agent: 'Frequency Analyzer', message: 'Detecting spatial frequency anomalies and latent blending edges...' },
  { agent: 'GAN Authenticator', message: 'Verifying photon-response non-uniformity (PRNU) sensor noise...' },
  { agent: 'Compression Check', message: 'Evaluating spatial block artifacts indicative of latent upscaling...' },
  { agent: 'Verdict Engine', message: 'Aggregating ensemble votes across temporal and spatial layers...' },
  { agent: 'Log Formatter', message: 'Constructing deterministic explanation mapping...' },
  { agent: 'System Core', message: 'Video analysis payload complete and verified.' }
];

const VideoAnalysis = () => {
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
      
      const totalMs = 45000;
      const tickMs = 200;
      let elapsed = 0;
      
      intervalRef.current = setInterval(() => {
        elapsed += tickMs;
        const pct = Math.min((elapsed / totalMs) * 100, 99);
        setProgress(pct);
        if (elapsed >= totalMs) clearInterval(intervalRef.current);
      }, tickMs);
      
      const stageMs = totalMs / VIDEO_STAGES.length;
      let si = 0;
      stageIntervalRef.current = setInterval(() => {
        si++;
        if (si < VIDEO_STAGES.length) {
          setStageIndex(si);
        } else {
          clearInterval(stageIntervalRef.current);
        }
      }, stageMs);

      const startTime = Date.now();
      try {
        const data = await analyzeVideo(file);
        const remaining = Math.max(0, totalMs - (Date.now() - startTime));
        await new Promise(r => setTimeout(r, remaining));
        
        clearInterval(intervalRef.current);
        clearInterval(stageIntervalRef.current);
        setProgress(100);
        setStageIndex(VIDEO_STAGES.length - 1);
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
          <h1 className="text-3xl font-heading font-bold text-gray-900">Deepfake Video Detection</h1>
          <p className="text-gray-500 mt-1">Advanced temporal and frequency analysis for video content.</p>
        </header>

        {!loading && !result && (
          <div className="bg-white p-12 rounded-[2rem] border-2 border-dashed border-gray-200 flex flex-col items-center justify-center transition-colors hover:border-blue-300">
            {!file ? (
              <div className="text-center">
                <div className="w-24 h-24 bg-blue-50 text-blue-600 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-sm">
                  <VideoIcon size={40} />
                </div>
                <h3 className="text-2xl font-bold font-heading mb-4">Upload Video for Analysis</h3>
                <p className="text-gray-400 mb-8 max-w-sm mx-auto">Supports MP4, AVI and MOV. Processing takes about 15-30 seconds.</p>
                <label className="bg-blue-600 text-white px-8 py-4 rounded-xl font-bold cursor-pointer inline-flex items-center gap-2 hover:bg-blue-700 transition-all active:scale-95 shadow-lg shadow-blue-200">
                  <input type="file" className="hidden" accept="video/*" onChange={onFileChange} />
                  Choose Video File
                </label>
              </div>
            ) : (
              <div className="w-full max-w-2xl">
                <div className="relative rounded-2xl overflow-hidden shadow-2xl mb-8 bg-black aspect-video flex items-center justify-center">
                  <video src={preview!} className="max-h-full" controls />
                  <button onClick={reset} className="absolute top-4 right-4 p-2 bg-white/20 hover:bg-white/40 backdrop-blur rounded-full text-white transition-colors">
                      <X size={20} />
                  </button>
                </div>
                
                <div className="flex flex-col gap-6">
                  <div className="flex justify-between items-center bg-gray-50 p-4 rounded-xl border border-gray-100">
                     <div className="flex items-center gap-3">
                         <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-sm">
                             <VideoIcon size={20} className="text-blue-500" />
                         </div>
                         <div className="max-w-[300px]">
                             <p className="text-sm font-bold text-gray-900 truncate">{file.name}</p>
                             <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                         </div>
                     </div>
                  </div>

                  <button onClick={handleUpload} className="bg-blue-600 text-white w-full py-4 rounded-xl font-bold text-lg shadow-lg shadow-blue-200">
                      Start Deepfake Scan
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {loading && (
            <AnalysisLogs stages={VIDEO_STAGES} progress={progress} stageIndex={stageIndex} fileName={file?.name} />
        )}

        {result && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
            <div className={`p-8 rounded-[2rem] border-4 flex items-center justify-between ${
                result.score >= 60 ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100'
            }`}>
                <div className="flex items-center gap-6">
                    <div className={`w-20 h-20 rounded-2xl flex items-center justify-center ${
                         result.score >= 60 ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                    }`}>
                        {result.score >= 60 ? <AlertCircle size={40} /> : <ShieldCheck size={40} />}
                    </div>
                    <div>
                        <h2 className={`text-3xl font-heading font-bold ${result.score >= 60 ? 'text-red-900' : 'text-green-900'}`}>
                            {result.verdict}
                        </h2>
                        <p className={`font-medium ${result.score >= 60 ? 'text-red-600' : 'text-green-600'}`}>
                            Overall fake probability: <span className="font-bold">{result.score}%</span>
                        </p>
                    </div>
                </div>
                <button onClick={reset} className="btn-secondary">Analyze Another</button>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
                {[
                    { title: 'Temporal Consistency', value: result.details.temporal_score.toFixed(2), desc: 'Stability of features between sequential frames.', icon: History, rating: result.details.temporal_score < 2 ? 'High Risk' : 'Healthy' },
                    { title: 'Spectral Analysis', value: result.details.frequency_score.toFixed(2), desc: 'Artifact detection in frequency domain across frames.', icon: Activity, rating: result.details.frequency_score > 6 ? 'Suspicious' : 'Clean' },
                    { title: 'Joint Votes', value: `${result.details.fake_votes} / 12`, desc: 'Ensemble agreement between dual AI detection layers.', icon: TrendingUp, rating: result.details.fake_votes > 4 ? 'High Fake Signal' : 'Low Variance' }
                ].map((stat, i) => (
                    <div key={i} className="bg-white p-8 rounded-3xl border border-gray-100 shadow-sm">
                        <div className="flex items-center gap-3 mb-6">
                            <stat.icon className="text-blue-600" size={24} />
                            <h3 className="font-bold text-gray-900">{stat.title}</h3>
                        </div>
                        <p className="text-4xl font-heading font-bold mb-4">{stat.value}</p>
                        <p className="text-sm text-gray-500 leading-relaxed mb-6">{stat.desc}</p>
                        <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                            stat.rating.includes('Risk') || stat.rating.includes('Suspicious') || stat.rating.includes('High')
                            ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                        }`}>
                            {stat.rating}
                        </div>
                    </div>
                ))}
            </div>

            <div className="bg-white p-8 rounded-3xl border border-gray-100">
                <h3 className="text-xl font-heading font-bold mb-6">Processing Details</h3>
                <div className="space-y-4">
                    <p className="text-gray-600">The video was processed using a multi-layer detection pipeline. 
                    Temporal analysis examined the frame-to-frame flow of facial features and textures. 
                    Simultaneously, frequency analysis sought out common upsampling artifacts present in most deepfake generation pipelines.</p>
                    <div className="flex gap-4 pt-4">
                        <div className="w-1/2 h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500" style={{width: '100%'}}></div>
                        </div>
                        <div className="w-1/2 h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500" style={{width: '80%'}}></div>
                        </div>
                    </div>
                </div>
            </div>
          </motion.div>
        )}
      </div>
    );
};

export default VideoAnalysis;
