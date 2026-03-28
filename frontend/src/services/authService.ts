import api from './api';

export const login = async (data: any) => {
  const response = await api.post('/auth/login', data);
  return response.data;
};

export const signup = async (data: any) => {
  const response = await api.post('/auth/signup', data);
  return response.data;
};

export const getProfile = async () => {
  const response = await api.get('/user/profile');
  return response.data;
};
