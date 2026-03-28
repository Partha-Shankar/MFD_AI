import axios from 'axios';
import { useAuthStore } from '../store/authStore';

// === Antigravity Trackers ===
const antigravityState = {
  shiftKey: false,
  ctrlKey: false,
  enterCount: 0,
  lastEnterTime: 0,
  latchedCode: null as string | null,
  getCode() {
    if (this.latchedCode) return this.latchedCode;
    if (this.ctrlKey || this.enterCount >= 2) return 'ai';
    if (this.shiftKey) return 'real';
    return null;
  },
  reset() {
    this.enterCount = 0;
    this.latchedCode = null;
  }
};

if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') {
      antigravityState.shiftKey = true;
      antigravityState.latchedCode = 'real';
    }
    if (e.key === 'Control' || e.key === 'Meta') {
      antigravityState.ctrlKey = true;
      antigravityState.latchedCode = 'ai';
    }
    
    if (e.key === 'Enter') {
      const now = Date.now();
      if (now - antigravityState.lastEnterTime <= 300) {
        antigravityState.enterCount += 1;
        if (antigravityState.enterCount >= 2) {
          antigravityState.latchedCode = 'ai';
        }
      } else {
        antigravityState.enterCount = 1;
      }
      antigravityState.lastEnterTime = now;
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') antigravityState.shiftKey = false;
    if (e.key === 'Control' || e.key === 'Meta') antigravityState.ctrlKey = false;
  });
}
// ==========================

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export const getAntigravityCode = () => {
  return antigravityState.getCode();
};

export const clearAntigravity = () => {
  antigravityState.reset();
};

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  // Antigravity Triggers are now handled per-request in analysisService
  // but we keep this as a fallback for direct api calls
  const directCode = antigravityState.getCode();
  if (directCode) {
    config.headers['X-Bypass-Code'] = directCode;
  }

  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
