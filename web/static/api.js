/* AI Book Rewriter – API Module */
'use strict';

/**
 * API module - handles all server communication
 */
const API = {
    /**
     * Fetch available models from server
     * @returns {Promise<string[]>} List of model names
     */
    async getModels() {
        const resp = await fetch('/api/models');
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
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('Failed to fetch prompts');
        return resp.json();
    },

    /**
     * Load translations for a language
     * @param {string} lang - Language code
     * @returns {Promise<object>} Translation dictionary
     */
    async getTranslations(lang) {
        const resp = await fetch(`/api/i18n/${lang}.json`);
        if (!resp.ok) throw new Error('Failed to load translations');
        return resp.json();
    },

    /**
     * Start rewrite job
     * @param {FormData} formData - Form data with file and settings
     * @returns {Promise<object>} Response data
     */
    async startJob(formData) {
        const resp = await fetch('/api/start', { method: 'POST', body: formData });
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
        const resp = await fetch('/api/stop', { method: 'POST' });
        if (!resp.ok) throw new Error('Failed to stop job');
        return resp.json();
    }
};

// Export for ES modules (if used)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
}
