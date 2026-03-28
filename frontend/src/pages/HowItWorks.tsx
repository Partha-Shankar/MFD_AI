import React from 'react';
import { motion } from 'framer-motion';
import { 
    Upload, 
    Cpu, 
    Activity, 
    Search, 
    ShieldCheck, 
    FileText,
    ArrowRight
} from 'lucide-react';

const HowItWorks = () => {
    const steps = [
        {
            title: 'Upload Content',
            desc: 'Securely upload your image (JPG/PNG) or video (MP4/AVI/MOV) to our cloud processing unit.',
            icon: Upload,
            color: 'bg-blue-50 text-blue-600'
        },
        {
            title: 'AI Neural Scanning',
            desc: 'State-of-the-art transformer models perform a semantic analysis of the visual content.',
            icon: Cpu,
            color: 'bg-purple-50 text-purple-600'
        },
        {
            title: 'Frequency Domain Check',
            desc: 'We look for spectral artifacts and GAN-specific patterns in the frequency domain.',
            icon: Activity,
            color: 'bg-emerald-50 text-emerald-600'
        },
        {
            title: 'Texture & Noise Mapping',
            desc: 'Authentic sensor noise patterns are compared against suspicious low-complexity areas.',
            icon: Search,
            color: 'bg-amber-50 text-amber-600'
        },
        {
            title: 'Ensemble Verification',
            desc: 'Multiple detectors vote on the veracity of the sample to minimize false positives.',
            icon: ShieldCheck,
            color: 'bg-primary-50 text-primary-600'
        },
        {
            title: 'Detailed Reporting',
            desc: 'Get a comprehensive breakdown of findings with technical scores for every detection layer.',
            icon: FileText,
            color: 'bg-rose-50 text-rose-600'
        }
    ];

    return (
        <div className="max-w-5xl mx-auto space-y-12 pb-20">
            <header className="text-center space-y-4">
                <h1 className="text-4xl font-heading font-bold text-gray-900">How Our Technology Works</h1>
                <p className="text-xl text-gray-500 max-w-2xl mx-auto">
                    A multi-layered approach to digital forensics and fake content detection.
                </p>
            </header>

            <div className="relative pt-10">
                {/* Connection line */}
                <div className="absolute left-[50%] top-0 bottom-0 w-1 bg-gray-100 hidden md:block -translate-x-1/2 -z-10" />
                <div className="absolute left-[50%] top-0 h-10 w-1 bg-gradient-to-t from-gray-100 to-transparent hidden md:block -translate-x-1/2 -z-10" />

                <div className="space-y-12 md:space-y-0">
                    {steps.map((step, i) => (
                        <motion.div 
                            key={i}
                            initial={{ opacity: 0, x: i % 2 === 0 ? -50 : 50 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            className={`flex flex-col md:flex-row items-center gap-8 ${
                                i % 2 === 1 ? 'md:flex-row-reverse' : ''
                            } md:mb-24`}
                        >
                            <div className="md:w-1/2 flex flex-col items-center md:items-start text-center md:text-left px-6">
                                <span className="text-xs font-bold text-primary-600 uppercase tracking-widest mb-4">Step {i + 1}</span>
                                <h3 className="text-2xl font-bold font-heading mb-4 text-gray-900">{step.title}</h3>
                                <p className="text-gray-500 leading-relaxed text-lg">{step.desc}</p>
                            </div>

                            <div className="relative flex items-center justify-center">
                                <div className="absolute w-24 h-24 bg-white rounded-full border-4 border-gray-50 -z-10 shadow-sm hidden md:block" />
                                <div className={`w-20 h-20 ${step.color} rounded-3xl flex items-center justify-center shadow-xl shadow-gray-200/50 transform rotate-12 transition-transform hover:rotate-0 duration-500`}>
                                    <step.icon size={32} />
                                </div>
                            </div>

                            <div className="md:w-1/2 hidden md:block" />
                        </motion.div>
                    ))}
                </div>
            </div>

            <div className="bg-primary-900 rounded-[3rem] p-12 text-white overflow-hidden relative">
                <div className="absolute top-0 right-0 p-12 opacity-10">
                    <Cpu size={240} />
                </div>
                <div className="relative z-10 text-center max-w-2xl mx-auto">
                    <h2 className="text-3xl font-heading font-bold mb-6">Ready to secure your content?</h2>
                    <p className="text-primary-100 text-lg mb-10">
                        Join thousands of researchers and organizations using MFD AI to fight disinformation.
                    </p>
                    <div className="flex justify-center gap-4">
                        <button className="px-8 py-4 bg-white text-primary-900 rounded-xl font-bold hover:bg-primary-50 transition-colors shadow-xl">
                            Start Scanning Now
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HowItWorks;
