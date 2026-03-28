import React from 'react';
import { motion } from 'framer-motion';
import { Target, Users, BookOpen, Globe } from 'lucide-react';

const About = () => {
    return (
        <div className="max-w-6xl mx-auto space-y-20 pb-20">
            <header className="relative py-20 bg-primary-600 rounded-[3rem] overflow-hidden text-center text-white px-6">
                <div className="absolute inset-0 bg-gradient-to-tr from-primary-900/50 to-transparent"></div>
                <div className="relative z-10 max-w-3xl mx-auto">
                    <h1 className="text-5xl font-heading font-bold mb-6">Securing the Truth in an Age of AI</h1>
                    <p className="text-xl text-primary-100 leading-relaxed">
                        Multimodal Fake Content Detector (MFD AI) provides advanced tools to verify the authenticity of digital media.
                    </p>
                </div>
            </header>

            <div className="grid md:grid-cols-2 gap-20 items-center">
                <div className="space-y-6">
                    <h2 className="text-4xl font-heading font-bold text-gray-900 leading-tight">Our Mission</h2>
                    <p className="text-lg text-gray-600 leading-relaxed">
                        Disinformation and synthetic media are becoming increasingly sophisticated. Our mission is to build an open, accessible, and highly accurate platform that anyone can use to check if an image or video is legitimate.
                    </p>
                    <p className="text-lg text-gray-600 leading-relaxed">
                        We believe that trust is the foundation of digital interaction. By providing state-of-the-art forensic tools, we empower individuals, journalists, and organizations to combat fake news and deepfakes.
                    </p>
                    <div className="flex gap-12 pt-6">
                        <div>
                            <p className="text-4xl font-bold font-heading text-primary-600">10M+</p>
                            <p className="text-sm font-bold text-gray-400 uppercase tracking-widest mt-1">Files Analyzed</p>
                        </div>
                        <div>
                            <p className="text-4xl font-bold font-heading text-primary-600">99%</p>
                            <p className="text-sm font-bold text-gray-400 uppercase tracking-widest mt-1">Success Rate</p>
                        </div>
                    </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-4 pt-12">
                        <div className="bg-white p-6 rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 flex flex-col gap-4">
                            <Target className="text-blue-500" size={32} />
                            <h3 className="font-bold text-gray-900">Precision</h3>
                        </div>
                        <div className="bg-white p-6 rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 flex flex-col gap-4">
                            <BookOpen className="text-purple-500" size={32} />
                            <h3 className="font-bold text-gray-900">Research</h3>
                        </div>
                    </div>
                    <div className="space-y-4">
                        <div className="bg-white p-6 rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 flex flex-col gap-4">
                            <Users className="text-emerald-500" size={32} />
                            <h3 className="font-bold text-gray-900">Community</h3>
                        </div>
                        <div className="bg-white p-6 rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 flex flex-col gap-4">
                            <Globe className="text-amber-500" size={32} />
                            <h3 className="font-bold text-gray-900">Impact</h3>
                        </div>
                    </div>
                </div>
            </div>

            <section className="bg-gray-50 rounded-[3rem] p-16 border border-gray-100">
                <div className="max-w-3xl mx-auto text-center space-y-12">
                    <h2 className="text-3xl font-heading font-bold text-gray-900">AI Collaboration</h2>
                    <p className="text-lg text-gray-600 leading-relaxed">
                        MFD AI leverages research from top institutions including OpenAI, Transformers (Hugging Face), and specialized deepfake detection datasets. We continuously update our models to identify the latest generative patterns from Midjourney, DALL-E, and advanced video synthesis tools.
                    </p>
                    <div className="flex flex-wrap justify-center gap-8 grayscale opacity-50">
                        {/* Placeholder for tech logos */}
                        <span className="font-bold text-2xl tracking-tighter">TRANSFORMERS</span>
                        <span className="font-bold text-2xl tracking-tighter">PYTORCH</span>
                        <span className="font-bold text-2xl tracking-tighter">OPENCV</span>
                        <span className="font-bold text-2xl tracking-tighter">FASTAPI</span>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default About;
