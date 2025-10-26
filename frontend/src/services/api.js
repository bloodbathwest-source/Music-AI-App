import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add token to requests
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  // Authentication
  async register(email, username, password) {
    const response = await this.client.post('/auth/register', {
      email,
      username,
      password,
    });
    return response.data;
  }

  async login(username, password) {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await this.client.post('/auth/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  async getCurrentUser() {
    const response = await this.client.get('/auth/me');
    return response.data;
  }

  // Music Generation
  async generateMusic(params) {
    const response = await this.client.post('/music/generate', params);
    return response.data;
  }

  async getGenerationStatus(taskId) {
    const response = await this.client.get(`/music/generate/${taskId}`);
    return response.data;
  }

  async listTracks(skip = 0, limit = 10) {
    const response = await this.client.get('/music/tracks', {
      params: { skip, limit },
    });
    return response.data;
  }

  async getTrack(trackId) {
    const response = await this.client.get(`/music/tracks/${trackId}`);
    return response.data;
  }

  async deleteTrack(trackId) {
    const response = await this.client.delete(`/music/tracks/${trackId}`);
    return response.data;
  }

  async shareTrack(trackId, isPublic) {
    const response = await this.client.post(`/music/tracks/${trackId}/share`, {
      is_public: isPublic,
    });
    return response.data;
  }

  // Export
  async exportTrack(trackId, format, quality = 'high') {
    const response = await this.client.post('/export/export', {
      track_id: trackId,
      format,
      quality,
    });
    return response.data;
  }

  async uploadToPlatform(trackId, platform, title, description, genre, tags) {
    const response = await this.client.post('/export/upload', {
      track_id: trackId,
      platform,
      title,
      description,
      genre,
      tags,
    });
    return response.data;
  }

  async getUploadStatus(uploadId) {
    const response = await this.client.get(`/export/upload/${uploadId}/status`);
    return response.data;
  }

  // User
  async getUserProfile() {
    const response = await this.client.get('/users/profile');
    return response.data;
  }

  async updateUserProfile(data) {
    const response = await this.client.put('/users/profile', data);
    return response.data;
  }
}

export default new ApiService();
