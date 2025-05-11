// static/js/api.js
import { getToken, clearTokenAndReload } from './auth.js';
import { API_BASE_URL } from './config.js';

export async function fetchWithAuth(endpoint, options = {}) {
    const token = getToken();
    const headers = { ...options.headers };

    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, { ...options, headers });

    if (response.status === 401 && endpoint !== '/login' && endpoint !== '/signup') { // Don't clear on login/signup fail
        console.warn(`API request to ${endpoint} unauthorized. Clearing token and reloading.`);
        clearTokenAndReload();
        throw new Error('Unauthorized');
    }
    return response;
}

export async function loginUser(username, password) {
    const response = await fetchWithAuth('/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Login failed due to network or server error.' }));
        throw new Error(errorData.detail || 'Login failed');
    }
    return response.json();
}

export async function signupUser(username, password) {
    const response = await fetchWithAuth('/signup', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
    });
     if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Signup failed due to network or server error.' }));
        throw new Error(errorData.detail || 'Signup failed');
    }
    return response.json();
}

export async function fetchConversations() {
    const response = await fetchWithAuth('/conversations');
    if (!response.ok) throw new Error('Failed to fetch conversations');
    return response.json();
}

export async function fetchConversationMessages(conversationId) {
    const response = await fetchWithAuth(`/conversations/${conversationId}`);
    if (!response.ok) throw new Error('Failed to fetch conversation messages');
    return response.json();
}

export async function fetchOllamaModels() {
    const response = await fetchWithAuth('/ollama-models');
    if (!response.ok) throw new Error('Failed to fetch Ollama models');
    return response.json();
}

export async function fetchOpenAIModels() {
    const response = await fetchWithAuth('/openai-models');
    if (!response.ok) throw new Error('Failed to fetch OpenAI models');
    return response.json();
}

export async function fetchRequestProfiles() {
    const response = await fetchWithAuth('/request-profiles');
    if (!response.ok) throw new Error('Failed to fetch request profiles');
    return response.json();
}

export async function forkConversationApi(originalConversationId, messageIndex) {
    const response = await fetchWithAuth('/conversations/fork', {
        method: 'POST',
        body: JSON.stringify({
            original_conversation_id: originalConversationId,
            message_index: messageIndex,
        }),
    });
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Forking failed due to network or server error.' }));
        throw new Error(errorData.detail || 'Forking failed');
    }
    return response.json();
}