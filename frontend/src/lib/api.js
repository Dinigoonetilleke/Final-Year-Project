const BASE_URL = 'http://127.0.0.1:5000/api';

async function handleResponse(response) {
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
}

async function request(path, options = {}) {
  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });

    return await handleResponse(response);
  } catch (error) {
    if (error.message === 'Failed to fetch') {
      throw new Error('Cannot connect to backend. Is the server running?');
    }
    throw error;
  }
}

export const api = {
  get(path) {
    return request(path);
  },

  post(path, body) {
    return request(path, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },

  put(path, body) {
    return request(path, {
      method: 'PUT',
      body: JSON.stringify(body),
    });
  },

  delete(path) {
    return request(path, {
      method: 'DELETE',
    });
  },
};