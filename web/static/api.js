/* AI Book Rewriter – API Module */
'use strict';

/**
 * API module - handles all server communication
 */
const API_TIMEOUT_MS = 30000;

function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
}

async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
    const headers = new Headers(options.headers || {});

    if ((options.method || 'GET').toUpperCase() === 'POST') {
        headers.set('X-CSRF-Token', getCsrfToken());
    }

    try {
        return await fetch(url, { ...options, headers, signal: controller.signal });
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Request timed out');
        }
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

const API = {
    /**
     * Fetch available models from server
     * @returns {Promise<string[]>} List of model names
     */
    async getModels() {
        const resp = await fetchWithTimeout('/api/models');
        if (!resp.ok) throw new Error('Failed to fetch models');
        return resp.json();
    },

    /**
     * Fetch available prompts
     * @param {string} lang - Language code for prompt descriptions
     * @returns {Promise<Array>} List of prompt objects
     */
    async getPrompts(lang = '') {
        const url = '/api/prompts' + (lang ? '?lang=' + lang : '');
        const resp = await fetchWithTimeout(url);
        if (!resp.ok) throw new Error('Failed to fetch prompts');
        return resp.json();
    },

    async getPromptCategories(lang = '') {
        const url = '/api/prompts/categories' + (lang ? '?lang=' + lang : '');
        const resp = await fetchWithTimeout(url);
        if (!resp.ok) throw new Error('Failed to fetch prompt categories');
        return resp.json();
    },

    /**
     * Load translations for a language
     * @param {string} lang - Language code
     * @returns {Promise<object>} Translation dictionary
     */
    async getTranslations(lang) {
        const resp = await fetchWithTimeout(`/api/i18n/${lang}.json`);
        if (!resp.ok) throw new Error('Failed to load translations');
        return resp.json();
    },

    /**
     * Start rewrite job
     * @param {FormData} formData - Form data with file and settings
     * @returns {Promise<object>} Response data
     */
    async startJob(formData) {
        const resp = await fetchWithTimeout('/api/start', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) {
            const error = new Error(data.error || resp.statusText);
            error.data = data;
            throw error;
        }
        return data;
    },

    /**
     * Stop current job
     * @returns {Promise<void>}
     */
    async stopJob() {
        const resp = await fetchWithTimeout('/api/stop', { method: 'POST' });
        if (!resp.ok) throw new Error('Failed to stop job');
        return resp.json();
    },

    async saveSettings(settings) {
        const resp = await fetchWithTimeout('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        const data = await resp.json();
        if (!resp.ok) {
            const error = new Error(data.error || resp.statusText);
            error.data = data;
            throw error;
        }
        return data;
    }
};

// Export for ES modules (if used)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}
