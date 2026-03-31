/* AI Book Rewriter – Web Frontend (Main Module) */
'use strict';

// ── Toast Notifications ─────────────────────────────────────────────────────
const ToastManager = {
    container: null,
    
    init() {
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            document.body.appendChild(this.container);
        }
    },
    
    /**
     * Show a toast notification
     * @param {string} message - Message to display
     * @param {string} type - Type: 'success', 'error', 'warning', 'info'
     * @param {number} duration - Duration in ms (default: 4000)
     */
    show(message, type = 'info', duration = 4000) {
        if (!this.container) this.init();
        
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };
        
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || 'ℹ'}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" aria-label="Close">&times;</button>
        `;
        
        this.container.appendChild(toast);
        
        // Trigger animation
        requestAnimationFrame(() => toast.classList.add('toast-show'));
        
        // Close button
        toast.querySelector('.toast-close').addEventListener('click', () => {
            this._remove(toast);
        });
        
        // Auto-remove
        if (duration > 0) {
            setTimeout(() => this._remove(toast), duration);
        }
        
        return toast;
    },
    
    _remove(toast) {
        toast.classList.remove('toast-show');
        toast.classList.add('toast-hide');
        setTimeout(() => toast.remove(), 300);
    },
    
    success(message, duration) { return this.show(message, 'success', duration); },
    error(message, duration) { return this.show(message, 'error', duration); },
    warning(message, duration) { return this.show(message, 'warning', duration); },
    info(message, duration) { return this.show(message, 'info', duration); }
};

// ── Skeleton Loading ────────────────────────────────────────────────────────
const SkeletonManager = {
    /**
     * Show skeleton on an element
     * @param {HTMLElement} element
     */
    show(element) {
        if (!element) return;
        element.classList.add('skeleton-loading');
        element.setAttribute('data-skeleton', 'true');
    },
    
    /**
     * Hide skeleton from an element
     * @param {HTMLElement} element
     */
    hide(element) {
        if (!element) return;
        element.classList.remove('skeleton-loading');
        element.removeAttribute('data-skeleton');
    },
    
    /**
     * Show skeleton on select element (replace options with skeleton)
     * @param {HTMLSelectElement} select
     */
    showSelectSkeleton(select) {
        if (!select) return;
        select.classList.add('skeleton-select');
        select.innerHTML = '<option value="" class="skeleton-option">████████</option>';
    },
    
    /**
     * Hide skeleton from select element
     * @param {HTMLSelectElement} select
     */
    hideSelectSkeleton(select) {
        if (!select) return;
        select.classList.remove('skeleton-select');
    }
};

// ── Form Validation ──────────────────────────────────────────────────────────
const FormValidator = {
    /**
     * Validate file input
     * @param {HTMLInputElement} input
     * @returns {object} { valid, message }
     */
    validateFileInput(input) {
        if (!input) return { valid: false, message: 'Input not found' };
        
        const file = input.files[0];
        if (!file) {
            return { valid: false, message: ViewModel.t('validation_file_required') || 'Please select a file' };
        }
        
        if (!file.name.endsWith('.txt')) {
            return { valid: false, message: ViewModel.t('validation_file_type') || 'Only .txt files are allowed' };
        }
        
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            return { valid: false, message: ViewModel.t('validation_file_size') || 'File size must be under 10MB' };
        }
        
        return { valid: true, message: '' };
    },
    
    /**
     * Validate form field and show feedback
     * @param {HTMLElement} field
     * @param {string} type - Field type
     */
    validateField(field, type) {
        let result = { valid: true, message: '' };
        
        switch (type) {
            case 'file':
                result = this.validateFileInput(field);
                break;
            case 'select':
                result = { valid: !!field.value, message: field.value ? '' : 'Please select an option' };
                break;
        }
        
        ViewModel.setValidation(type, result.valid, result.message);
        this.showFieldFeedback(field, result);
        
        return result.valid;
    },
    
    /**
     * Show visual feedback on field
     * @param {HTMLElement} field
     * @param {object} result
     */
    showFieldFeedback(field, result) {
        const wrapper = field.closest('.form-row') || field.parentElement;
        
        // Remove existing feedback
        const existing = wrapper.querySelector('.field-feedback');
        if (existing) existing.remove();
        
        // Update field classes
        field.classList.remove('field-valid', 'field-invalid');
        if (!result.valid) {
            field.classList.add('field-invalid');
            
            // Add feedback message
            const feedback = document.createElement('span');
            feedback.className = 'field-feedback field-error';
            feedback.textContent = result.message;
            wrapper.appendChild(feedback);
        } else if (field.value || field.files?.length) {
            field.classList.add('field-valid');
        }
    },
    
    /**
     * Validate entire form
     * @param {HTMLFormElement} form
     * @returns {boolean}
     */
    validateForm(form) {
        const fileInput = form.querySelector('#input_file');
        const languageSelect = form.querySelector('#language');
        const modelSelect = form.querySelector('#model');
        
        const fileValid = this.validateField(fileInput, 'file');
        const langValid = this.validateField(languageSelect, 'select');
        const modelValid = this.validateField(modelSelect, 'select');
        
        return fileValid && langValid && modelValid;
    }
};

// ── SSE Connection ──────────────────────────────────────────────────────────
let evtSource = null;

function connectSSE() {
    if (evtSource) evtSource.close();
    evtSource = new EventSource('/events');

    evtSource.addEventListener('status', e => {
        const data = JSON.parse(e.data);
        syncStatus(data);
    });
    
    evtSource.addEventListener('progress', e => {
        const d = JSON.parse(e.data);
        ViewModel.setProgress(d.current, d.total, d.pct);
    });
    
    evtSource.addEventListener('log', e => {
        const d = JSON.parse(e.data);
        appendLog(d.msg);
    });
    
    evtSource.addEventListener('done', e => {
        const d = JSON.parse(e.data);
        onDone(d.output);
    });
    
    evtSource.onopen = () => {
        ViewModel.setConnected(true);
    };
    
    evtSource.onerror = () => {
        ViewModel.setConnected(false);
        setTimeout(connectSSE, 3000);
    };
}

// ── Status sync ─────────────────────────────────────────────────────────────
function syncStatus(data) {
    const running = data.running;
    ViewModel.setRunning(running);
    
    if (data.total > 0) {
        ViewModel.setProgress(data.current, data.total, (data.current / data.total * 100));
    }
    
    if (!running && data.output) {
        ViewModel.setOutputFilename(data.output);
    }
}

// ── Log ───────────────────────────────────────────────────────────────────────
function appendLog(msg) {
    const area = document.getElementById('log-area');
    if (!area) return;
    
    const div = document.createElement('div');
    div.className = 'log-entry';
    
    if (/error|ошибка|критич/i.test(msg)) div.classList.add('err');
    else if (/warn|предупрежд/i.test(msg)) div.classList.add('warn');
    else if (/done|финал|завершено|ok/i.test(msg)) div.classList.add('ok');

    const ts = new Date().toLocaleTimeString();
    div.textContent = `[${ts}] ${msg}`;
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
}

function clearLog() {
    const area = document.getElementById('log-area');
    if (area) area.innerHTML = '';
}

// ── Done handler ─────────────────────────────────────────────────────────────
function onDone(output) {
    ViewModel.setRunning(false);
    if (output) {
        ViewModel.setOutputFilename(output);
        ToastManager.success(ViewModel.t('done') || 'Operation completed successfully!');
    }
    appendLog(ViewModel.t('done') || 'Done');
}

// ── Model refresh ─────────────────────────────────────────────────────────────
async function refreshModels() {
    const sel = document.getElementById('model');
    if (!sel) return;
    
    const cur = sel.value;
    SkeletonManager.showSelectSkeleton(sel);
    ViewModel.setLoading('models', true);
    
    try {
        const models = await API.getModels();
        ViewModel.setModels(models);
        
        sel.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = opt.textContent = m;
            if (m === cur) opt.selected = true;
            sel.appendChild(opt);
        });
        
        SkeletonManager.hideSelectSkeleton(sel);
        ToastManager.success(ViewModel.t('models_refreshed') || 'Models updated');
    } catch (e) {
        SkeletonManager.hideSelectSkeleton(sel);
        ToastManager.error(ViewModel.t('error_refresh_models') || 'Failed to refresh models');
        appendLog(ViewModel.t('error_refresh_models') || 'Error refreshing models');
    }
}

// ── Prompt management ─────────────────────────────────────────────────────────
async function loadPrompts(lang) {
    const sel = document.getElementById('prompt_preset');
    if (!sel) return;
    
    const cur = sel.value;
    SkeletonManager.showSelectSkeleton(sel);
    ViewModel.setLoading('prompts', true);
    
    try {
        const prompts = await API.getPrompts(lang);
        ViewModel.setPrompts(prompts);
        
        sel.innerHTML = '';
        prompts.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            if (p.id === cur) opt.selected = true;
            sel.appendChild(opt);
        });
        
        SkeletonManager.hideSelectSkeleton(sel);
        updatePromptPreview();
    } catch (e) {
        SkeletonManager.hideSelectSkeleton(sel);
        ToastManager.error(ViewModel.t('error_load_prompts') || 'Failed to load prompts');
    }
}

function updatePromptPreview() {
    const sel = document.getElementById('prompt_preset');
    const previewEl = document.getElementById('prompt-preview');
    if (!sel || !previewEl) return;

    const selectedId = sel.value;
    const prompt = ViewModel.getPromptById(selectedId);

    if (prompt) {
        previewEl.innerHTML = '<strong>' + prompt.name + '</strong><br>' +
            '<span class="prompt-desc">' + (prompt.description || '') + '</span>';
    } else {
        previewEl.innerHTML = '';
    }
}

function initPromptSelector() {
    const sel = document.getElementById('prompt_preset');
    if (sel) {
        sel.addEventListener('change', updatePromptPreview);
    }
    const langInput = document.querySelector('input[name="lang"]');
    const lang = langInput ? langInput.value : 'en';
    ViewModel.setLanguage(lang);
    loadPrompts(lang);
}

// ── Stop ──────────────────────────────────────────────────────────────────────
async function stopJob() {
    try {
        await API.stopJob();
        ToastManager.info(ViewModel.t('stop_requested') || 'Stop requested');
        appendLog(ViewModel.t('stop_requested') || 'Stop requested');
    } catch (e) {
        ToastManager.error(ViewModel.t('stop_error') || 'Failed to stop');
        appendLog(ViewModel.t('stop_error') + ': ' + e);
    }
}

// ── Suggest output filename ──────────────────────────────────────────────────
function suggestOutputName() {
    const fileInput = document.getElementById('input_file');
    const outputInput = document.getElementById('output_file');
    
    if (!fileInput || !outputInput) return;
    
    const file = fileInput.files[0];
    if (!file) {
        ToastManager.warning(ViewModel.t('select_input_first') || 'Please select input file first');
        return;
    }
    
    // Generate output filename based on input
    const inputName = file.name;
    const baseName = inputName.replace(/\.txt$/i, '');
    const outputName = baseName + '_final.txt';
    
    outputInput.value = outputName;
}

// ── Initialize i18n ──────────────────────────────────────────────────────────
async function initI18n() {
    const langInput = document.querySelector('input[name="lang"]');
    const lang = langInput ? langInput.value : 'en';
    ViewModel.setLoading('translations', true);
    
    try {
        const translations = await API.getTranslations(lang);
        ViewModel.setTranslations(translations);
        ViewModel.setLanguage(lang);
    } catch (e) {
        console.warn('Failed to load translations:', e);
    }
}

// ── UI Renderer (binds ViewModel to DOM) ─────────────────────────────────────
const UIRenderer = {
    init() {
        // Subscribe to ViewModel changes
        ViewModel.subscribe((state, changeType) => this.render(state, changeType));
    },
    
    render(state, changeType) {
        switch (changeType) {
            case 'connection':
                this.renderConnection(state.connected);
                break;
            case 'jobStatus':
                this.renderJobStatus(state.isRunning);
                break;
            case 'progress':
                this.renderProgress(state.progress);
                break;
            case 'output':
                this.renderOutput(state.outputFilename);
                break;
            case 'validation':
                // Handled by FormValidator
                break;
        }
    },
    
    renderConnection(connected) {
        const el = document.getElementById('conn-status');
        if (!el) return;
        el.textContent = connected 
            ? (ViewModel.t('connected') || 'Connected')
            : (ViewModel.t('disconnected') || 'Disconnected');
        el.className = 'badge ' + (connected ? 'badge-ok' : 'badge-err');
    },
    
    renderJobStatus(isRunning) {
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        if (startBtn) {
            startBtn.disabled = isRunning;
            startBtn.classList.toggle('btn-loading', false);
        }
        if (stopBtn) {
            stopBtn.disabled = !isRunning;
        }
    },
    
    renderProgress(progress) {
        const textEl = document.getElementById('progress-text');
        const pctEl = document.getElementById('progress-pct');
        const barEl = document.getElementById('progress-bar');
        
        if (textEl) {
            textEl.textContent = `${progress.current} / ${progress.total} ${ViewModel.t('blocks_processed') || 'blocks processed'}`;
        }
        if (pctEl) {
            pctEl.textContent = `${progress.percent.toFixed(1)}%`;
        }
        if (barEl) {
            barEl.style.width = `${progress.percent}%`;
        }
    },
    
    renderOutput(filename) {
        const row = document.getElementById('download-row');
        const link = document.getElementById('download-link');
        
        if (!filename) {
            if (row) row.style.display = 'none';
            return;
        }
        
        if (link) {
            link.href = `/api/download/${encodeURIComponent(filename)}`;
        }
        if (row) {
            row.style.display = 'block';
        }
    }
};

// ── Form submit ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    ToastManager.init();
    UIRenderer.init();
    
    await initI18n();

    // File name display
    const fileInput = document.getElementById('input_file');
    const fileLabel = document.getElementById('file-name');
    if (fileInput && fileLabel) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            fileLabel.textContent = file?.name || '—';
            
            // Validate file
            if (file) {
                FormValidator.validateField(fileInput, 'file');
            }
        });
    }

    // Form submit
    const form = document.getElementById('rewrite-form');
    if (form) {
        form.addEventListener('submit', async e => {
            e.preventDefault();
            
            // Validate form
            if (!FormValidator.validateForm(form)) {
                ToastManager.warning(ViewModel.t('validation_fix_errors') || 'Please fix the errors');
                return;
            }
            
            const fd = new FormData(form);
            // Ensure checkbox values are included
            if (!fd.get('resume')) fd.set('resume', 'false');
            else fd.set('resume', 'true');
            if (!fd.get('parallel_mode')) fd.set('parallel_mode', 'false');
            else fd.set('parallel_mode', 'true');

            // Show loading state
            const startBtn = document.getElementById('start-btn');
            if (startBtn) startBtn.classList.add('btn-loading');
            
            ViewModel.setRunning(true);
            document.getElementById('download-row').style.display = 'none';
            clearLog();
            ViewModel.setProgress(0, 0, 0);

            try {
                await API.startJob(fd);
                ToastManager.success(ViewModel.t('job_started') || 'Job started successfully');
            } catch (err) {
                ViewModel.setRunning(false);
                const errorMsg = err.data?.error || err.message;
                ToastManager.error(ViewModel.t('error') + ': ' + errorMsg);
                appendLog(ViewModel.t('error') + ': ' + errorMsg);
            }
        });
    }

    connectSSE();
    initPromptSelector();
});
