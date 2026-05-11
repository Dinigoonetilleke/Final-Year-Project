// Base URL of your backend
const BASE_URL = 'http://127.0.0.1:5000/api';

export const api = {
  // POST request (login, signup, evaluate, etc.)
  async post(path, body) {
    try {
      const response = await fetch(`${BASE_URL}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      // Convert response to JSON safely
      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error('Invalid server response');
      }

      // Handle errors
      if (!response.ok) {
        throw new Error(data.error || data.message || 'Request failed');
      }

      return data;
    } catch (error) {
      // This is what caused your "Failed to fetch"
      if (error.message === 'Failed to fetch') {
        throw new Error('Cannot connect to backend. Is the server running?');
      }
      throw error;
    }
  },

  // GET request (dashboard, reports, etc.)
  async get(path) {
    try {
      const response = await fetch(`${BASE_URL}${path}`);

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error('Invalid server response');
      }

      if (!response.ok) {
        throw new Error(data.error || data.message || 'Request failed');
      }

      return data;
    } catch (error) {
      if (error.message === 'Failed to fetch') {
        throw new Error('Cannot connect to backend. Is the server running?');
      }
      throw error;
    }
  },
};