/* AI Book Rewriter – View-Model Module */
'use strict';

/**
 * ViewModel - Central state management for UI
 * Separates business logic from DOM manipulation
 */
const ViewModel = {
    // ── State ─────────────────────────────────────────────────────────────
    state: {
        // Connection status
        connected: false,
        
        // Job status
        isRunning: false,
        jobId: null,
        
        // Progress
        progress: {
            current: 0,
            total: 0,
            percent: 0
        },
        
        // Data
        models: [],
        prompts: [],
        translations: {},
        currentLang: 'en',
        
        // Form validation
        validation: {
            inputFile: { valid: false, message: '' },
            language: { valid: true, message: '' },
            model: { valid: true, message: '' },
            promptPreset: { valid: true, message: '' }
        },
        
        // UI state
        isLoading: {
            models: false,
            prompts: false,
            translations: false
        },
        
        // Output
        outputFilename: null
    },

    // ── Subscribers for reactive updates ───────────────────────────────────
    _subscribers: [],

    /**
     * Subscribe to state changes
     * @param {Function} callback - Function to call on state change
     * @returns {Function} Unsubscribe function
     */
    subscribe(callback) {
        this._subscribers.push(callback);
        return () => {
            this._subscribers = this._subscribers.filter(cb => cb !== callback);
        };
    },

    /**
     * Notify all subscribers of state change
     * @param {string} changeType - Type of change for selective updates
     */
    _notify(changeType) {
        this._subscribers.forEach(cb => cb(this.state, changeType));
    },

    // ── State Update Methods ───────────────────────────────────────────────
    
    /**
     * Set connection status
     * @param {boolean} connected
     */
    setConnected(connected) {
        this.state.connected = connected;
        this._notify('connection');
    },

    /**
     * Set job running status
     * @param {boolean} isRunning
     */
    setRunning(isRunning) {
        this.state.isRunning = isRunning;
        if (!isRunning) {
            this.state.jobId = null;
        }
        this._notify('jobStatus');
    },

    /**
     * Update progress
     * @param {number} current
     * @param {number} total
     * @param {number} percent
     */
    setProgress(current, total, percent) {
        this.state.progress = { current, total, percent };
        this._notify('progress');
    },

    /**
     * Set models list
     * @param {string[]} models
     */
    setModels(models) {
        this.state.models = models;
        this.state.isLoading.models = false;
        this._notify('models');
    },

    /**
     * Set prompts list
     * @param {Array} prompts
     */
    setPrompts(prompts) {
        this.state.prompts = prompts;
        this.state.isLoading.prompts = false;
        this._notify('prompts');
    },

    /**
     * Set translations
     * @param {object} translations
     */
    setTranslations(translations) {
        this.state.translations = translations;
        this.state.isLoading.translations = false;
        this._notify('translations');
    },

    /**
     * Set current language
     * @param {string} lang
     */
    setLanguage(lang) {
        this.state.currentLang = lang;
        this._notify('language');
    },

    /**
     * Set loading state for a resource
     * @param {string} resource - Resource name (models, prompts, translations)
     * @param {boolean} loading
     */
    setLoading(resource, loading) {
        this.state.isLoading[resource] = loading;
        this._notify('loading');
    },

    /**
     * Set output filename for download
     * @param {string} filename
     */
    setOutputFilename(filename) {
        this.state.outputFilename = filename;
        this._notify('output');
    },

    /**
     * Update validation state for a field
     * @param {string} field - Field name
     * @param {boolean} valid
     * @param {string} message
     */
    setValidation(field, valid, message = '') {
        this.state.validation[field] = { valid, message };
        this._notify('validation');
    },

    /**
     * Check if form is valid
     * @returns {boolean}
     */
    isFormValid() {
        return Object.values(this.state.validation).every(v => v.valid);
    },

    // ── Translation Helper ─────────────────────────────────────────────────
    
    /**
     * Get translation for a key
     * @param {string} key - Translation key
     * @param {object} params - Parameters for interpolation
     * @returns {string}
     */
    t(key, params = {}) {
        let template = this.state.translations[key] || key;
        for (const [k, v] of Object.entries(params)) {
            template = template.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
        }
        return template;
    },

    /**
     * Get prompt by ID
     * @param {string} id
     * @returns {object|undefined}
     */
    getPromptById(id) {
        return this.state.prompts.find(p => p.id === id);
    }
};

// Export for ES modules (if used)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ViewModel;
}
