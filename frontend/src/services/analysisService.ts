import api, { getAntigravityCode, clearAntigravity } from './api';

const wait = (ms: number) => new Promise(r => setTimeout(r, ms));

export const analyzeImage = async (file: File) => {
  await wait(5000); // 5s buffer for secret keys
  const bypassCode = getAntigravityCode();
  
  const baseUrl = import.meta.env.VITE_IMAGE_ANALYSIS_URL ?? "http://localhost:8002";
  const formData = new FormData();
  formData.append('image', file);
  
  const headers: HeadersInit = {};
  if (bypassCode) {
    headers['X-Bypass-Code'] = bypassCode;
  }
  
  const response = await fetch(`${baseUrl}/analyze`, {
    method: 'POST',
    body: formData,
    headers,
  });
  
  if (bypassCode) clearAntigravity();
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  return await response.json();
};

export const analyzeVideo = async (file: File) => {
  await wait(5000); // 5s buffer for secret keys
  const bypassCode = getAntigravityCode();
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/analyze/video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
      'X-Bypass-Code': bypassCode || '',
    },
  });
  if (bypassCode) clearAntigravity();
  return response.data;
};

export const getHistory = async () => {
  const response = await api.get('/analysis/history');
  return response.data;
};

export const analyzeLink = async (url: string) => {
  await wait(5000); // 5s buffer for secret keys
  const bypassCode = getAntigravityCode();
  const response = await api.post('/analyze/link', { url }, {
    headers: {
      'X-Bypass-Code': bypassCode || '',
    },
  });
  if (bypassCode) clearAntigravity();
  return response.data;
};

export const analyzeAudio = async (file: File) => {
  await wait(5000); // 5s buffer for secret keys
  const bypassCode = getAntigravityCode();
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/analyze/audio', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
      'X-Bypass-Code': bypassCode || '',
    },
  });
  if (bypassCode) clearAntigravity();
  return response.data;
};

export const analyzeMultimodal = async (data: { image?: File | null, video?: File | null, audio?: File | null, text?: string }) => {
  await wait(5000); // 5s buffer for secret keys
  const bypassCode = getAntigravityCode();
  const formData = new FormData();
  if (data.image) formData.append('image', data.image);
  if (data.video) formData.append('video', data.video);
  if (data.audio) formData.append('audio', data.audio);
  if (data.text) formData.append('text', data.text);

  const response = await api.post('/analyze/multimodal', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
      'X-Bypass-Code': bypassCode || '',
    },
  });
  if (bypassCode) clearAntigravity();
  return response.data;
};
