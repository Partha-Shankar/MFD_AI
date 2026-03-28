import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import ImageAnalysis from './pages/ImageAnalysis';
import VideoAnalysis from './pages/VideoAnalysis';
import ResultsHistory from './pages/ResultsHistory';
import HowItWorks from './pages/HowItWorks';
import About from './pages/About';
import Profile from './pages/Profile';
import LinkAnalysis from './pages/LinkAnalysis';
import AudioAnalysis from './pages/AudioAnalysis';
import MultimodalAnalysis from './pages/MultimodalAnalysis';
import DashboardLayout from './components/layout/DashboardLayout';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        
        <Route element={<DashboardLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analyze-image" element={<ImageAnalysis />} />
          <Route path="/analyze-video" element={<VideoAnalysis />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/about" element={<About />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/analyze-link" element={<LinkAnalysis />} />
          <Route path="/analyze-audio" element={<AudioAnalysis />} />
          <Route path="/analyze-multimodal" element={<MultimodalAnalysis />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
