import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ShieldCheck, 
  Search, 
  Zap, 
  FileText, 
  Activity, 
  Lock,
  ArrowRight,
  Play
} from 'lucide-react';

const Landing = () => {
  return (
    <div className="bg-white text-gray-900 font-sans">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-primary-600 rounded-xl flex items-center justify-center shadow-lg shadow-primary-200">
              <ShieldCheck className="text-white" size={24} />
            </div>
            <span className="font-heading font-bold text-2xl tracking-tight">MFD AI</span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm font-medium text-gray-600 hover:text-primary-600 transition-colors">Features</a>
            <a href="#tech" className="text-sm font-medium text-gray-600 hover:text-primary-600 transition-colors">Technology</a>
            <a href="#how-it-works" className="text-sm font-medium text-gray-600 hover:text-primary-600 transition-colors">How it Works</a>
          </div>

          <div className="flex items-center gap-4">
            <Link to="/login" className="text-sm font-semibold text-gray-700 hover:text-primary-600 px-4 py-2 transition-colors">Login</Link>
            <Link to="/signup" className="btn-primary">Get Started</Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-40 pb-20 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="inline-block px-4 py-1.5 mb-6 text-sm font-bold tracking-wider text-primary-600 uppercase bg-primary-50 rounded-full">
              Advanced Deepfake Detection
            </span>
            <h1 className="text-5xl md:text-7xl font-heading font-bold mb-8 leading-tight tracking-tight">
              AI-Powered Detection for <br />
              <span className="gradient-text">Fake Images & Deepfake Videos</span>
            </h1>
            <p className="max-w-3xl mx-auto text-xl text-gray-600 mb-12 leading-relaxed">
              Protect your digital integrity with our multimodal analysis platform. We use state-of-the-art 
              AI models to detect structural anomalies, noise patterns, and semantic inconsistencies.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/signup" className="btn-primary flex items-center gap-2 text-lg px-8 py-4">
                Analyze Image <ArrowRight size={20} />
              </Link>
              <Link to="/signup" className="btn-secondary flex items-center gap-2 text-lg px-8 py-4">
                <Play size={20} fill="currentColor" /> Analyze Video
              </Link>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-20 relative"
          >
            <div className="glass-card overflow-hidden p-2 rounded-2xl shadow-2xl">
              <div className="bg-gray-50 rounded-xl max-w-5xl mx-auto aspect-video flex items-center justify-center border border-gray-100 relative group">
                <div className="absolute inset-0 bg-gradient-to-t from-black/5 to-transparent"></div>
                {/* Mock UI Preview */}
                <div className="w-full h-full p-8 flex flex-col">
                    <div className="flex justify-between items-center mb-12">
                        <div className="flex gap-2">
                            <div className="w-3 h-3 rounded-full bg-red-400"></div>
                            <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                            <div className="w-3 h-3 rounded-full bg-green-400"></div>
                        </div>
                        <div className="px-4 py-1.5 bg-white rounded-lg text-xs font-bold shadow-sm border border-gray-100">Live Detection Active</div>
                    </div>
                    <div className="grid grid-cols-3 gap-6 flex-1">
                        <div className="col-span-2 bg-white rounded-xl shadow-sm border border-gray-100 flex items-center justify-center flex-col gap-4">
                            <div className="w-20 h-20 bg-primary-50 rounded-full flex items-center justify-center">
                                <Search className="text-primary-600" size={32} />
                            </div>
                            <p className="text-gray-400 font-medium tracking-wide">AI SCANNING IN PROGRESS...</p>
                        </div>
                        <div className="flex flex-col gap-4">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex items-center gap-4">
                                    <div className="w-10 h-10 bg-gray-50 rounded-lg"></div>
                                    <div className="flex-1 space-y-2">
                                        <div className="w-full h-2 bg-gray-100 rounded"></div>
                                        <div className="w-2/3 h-2 bg-gray-50 rounded"></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
              </div>
            </div>
            {/* Floating Stats */}
            <div className="absolute -top-10 -right-10 bg-white p-6 rounded-2xl shadow-xl border border-gray-100 hidden lg:block transform rotate-3">
                <div className="flex items-center gap-4">
                   <div className="p-3 bg-green-50 text-green-600 rounded-xl">
                       <Zap size={24} />
                   </div>
                   <div>
                       <p className="text-xs font-bold text-gray-500 uppercase">Detection Rate</p>
                       <p className="text-2xl font-bold font-heading">99.8%</p>
                   </div>
                </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32 px-6 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-heading font-bold mb-6">Cutting-Edge Detection Layers</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Our system analyzes content across multiple dimensions to ensure the highest level of accuracy.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { icon: Activity, title: 'Frequency Artifacts', desc: 'Identifies spectral inconsistencies in images often left behind by GAN-based generative models.' },
              { icon: Search, title: 'Noise Pattern Analysis', desc: 'Detects artificial noise distributions that differ from natural camera sensor signatures.' },
              { icon: Zap, title: 'Structural Anomalies', desc: 'Analyzes edge complexity and pixel connectivity to find hidden manipulations.' },
              { icon: ShieldCheck, title: 'AI Image Detection', desc: 'State-of-the-art neural networks trained on millions of real vs fake samples.' },
              { icon: Lock, title: 'Authenticity Reporting', desc: 'Generates detailed scores and probability reports for legal and professional use.' },
              { icon: FileText, title: 'Deepfake Video Analysis', desc: 'Temporal consistency checks to identify frame-by-frame deepfake transitions.' },
            ].map((feature, idx) => (
              <motion.div 
                key={idx}
                whileHover={{ y: -10 }}
                className="bg-white p-10 rounded-3xl border border-gray-100 shadow-sm hover:shadow-xl transition-all"
              >
                <div className="w-16 h-16 bg-primary-50 text-primary-600 rounded-2xl flex items-center justify-center mb-8">
                  <feature.icon size={32} />
                </div>
                <h3 className="text-2xl font-bold mb-4 font-heading">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed text-lg">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-20 px-6 border-t border-gray-100">
        <div className="max-w-7xl mx-auto grid md:grid-cols-4 gap-12">
          <div className="col-span-2">
            <div className="flex items-center gap-2 mb-6">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <ShieldCheck className="text-white" size={18} />
              </div>
              <span className="font-heading font-bold text-xl tracking-tight">MFD AI</span>
            </div>
            <p className="text-gray-500 max-w-md leading-relaxed">
              Leading the way in multimodal fake content detection. Our platform empowers users to verify 
              digital content authenticity with AI precision.
            </p>
          </div>
          <div>
            <h4 className="font-bold mb-6 uppercase text-xs tracking-widest text-gray-400">Platform</h4>
            <ul className="space-y-4">
              <li><Link to="/about" className="text-gray-600 hover:text-primary-600">About Us</Link></li>
              <li><Link to="/how-it-works" className="text-gray-600 hover:text-primary-600">How It Works</Link></li>
              <li><Link to="/signup" className="text-gray-600 hover:text-primary-600">Careers</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-bold mb-6 uppercase text-xs tracking-widest text-gray-400">Legal</h4>
            <ul className="space-y-4">
              <li><Link to="/" className="text-gray-600 hover:text-primary-600">Privacy Policy</Link></li>
              <li><Link to="/" className="text-gray-600 hover:text-primary-600">Terms of Service</Link></li>
              <li><Link to="/" className="text-gray-600 hover:text-primary-600">Security</Link></li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto mt-20 pt-8 border-t border-gray-100 flex justify-between items-center text-sm text-gray-400">
          <p>© 2026 Multimodal Fake Detector. All rights reserved.</p>
          <div className="flex gap-8">
            <a href="#" className="hover:text-primary-600">Twitter</a>
            <a href="#" className="hover:text-primary-600">LinkedIn</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
