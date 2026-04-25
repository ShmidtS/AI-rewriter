/* AI Book Rewriter – Web Frontend (Main Module) */
'use strict';

(function () {

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
        
        const icon = document.createElement('span');
        icon.className = 'toast-icon';
        icon.textContent = icons[type] || 'ℹ';

        const messageEl = document.createElement('span');
        messageEl.className = 'toast-message';
        messageEl.textContent = message;

        const closeBtn = document.createElement('button');
        closeBtn.className = 'toast-close';
        closeBtn.setAttribute('aria-label', 'Close');
        closeBtn.textContent = '×';

        toast.append(icon, messageEl, closeBtn);
        this.container.appendChild(toast);

        // Trigger animation
        requestAnimationFrame(() => toast.classList.add('toast-show'));

        // Close button
        closeBtn.addEventListener('click', () => {
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
        select.replaceChildren();
        const option = document.createElement('option');
        option.value = '';
        option.className = 'skeleton-option';
        option.textContent = '████████';
        select.appendChild(option);
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
        const languageSelect = form.querySelector('#output_language');
        const modelSelect = form.querySelector('#model');
        
        const fileValid = this.validateField(fileInput, 'file');
        const langValid = this.validateField(languageSelect, 'select');
        const modelValid = this.validateField(modelSelect, 'select');
        
        return fileValid && langValid && modelValid;
    }
};

// ── SSE Connection ──────────────────────────────────────────────────────────
let evtSource = null;
let sseReconnectAttempts = 0;
let sseReconnectTimer = null;
const SSE_MAX_RECONNECT_DELAY = 30000; // max 30s between reconnects
const SSE_BASE_RECONNECT_DELAY = 3000;

function clearSseReconnectTimer() {
    if (sseReconnectTimer) {
        clearTimeout(sseReconnectTimer);
        sseReconnectTimer = null;
    }
}

function updateSseStatus(connected) {
    const el = document.getElementById('sse-status');
    if (!el) return;
    if (connected) {
        el.textContent = 'SSE: OK';
        el.className = 'badge badge-ok';
    } else {
        el.textContent = `SSE: Lost${sseReconnectAttempts > 0 ? ' (retry ' + sseReconnectAttempts + ')' : ''}`;
        el.className = 'badge badge-err';
    }
}

function closeSSE() {
    clearSseReconnectTimer();
    if (evtSource) {
        evtSource.close();
        evtSource = null;
    }
}

function connectSSE() {
    closeSSE();
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
        clearSseReconnectTimer();
        ViewModel.setConnected(true);
        updateSseStatus(true);
        sseReconnectAttempts = 0; // reset on successful connection
    };

    evtSource.onerror = () => {
        ViewModel.setConnected(false);
        updateSseStatus(false);
        sseReconnectAttempts++;
        // Exponential backoff: 3s, 6s, 12s, 24s, 30s cap
        const delay = Math.min(
            SSE_BASE_RECONNECT_DELAY * Math.pow(2, sseReconnectAttempts - 1),
            SSE_MAX_RECONNECT_DELAY
        );
        clearSseReconnectTimer();
        sseReconnectTimer = setTimeout(connectSSE, delay);
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
const MAX_LOG_ENTRIES = 500;

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
    while (area.children.length > MAX_LOG_ENTRIES) {
        area.firstElementChild?.remove();
    }
    area.scrollTop = area.scrollHeight;
}

function clearLog() {
    const area = document.getElementById('log-area');
    if (area) area.replaceChildren();
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
        
        sel.replaceChildren();
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
        
        sel.replaceChildren();
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

    previewEl.replaceChildren();
    if (prompt) {
        const nameEl = document.createElement('strong');
        nameEl.textContent = prompt.name;
        const lineBreak = document.createElement('br');
        const descEl = document.createElement('span');
        descEl.className = 'prompt-desc';
        descEl.textContent = prompt.description || '';
        previewEl.append(nameEl, lineBreak, descEl);
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

// ── Prompt Presets Card ─────────────────────────────────────────────────────

let allPresets = [];
let presetCategories = [];
let activeCategory = 'all';

async function loadPromptPresets() {
    const grid = document.getElementById('prompts-grid');
    const card = document.getElementById('prompts-card');
    if (!grid || !card) return;

    const langInput = document.querySelector('input[name="lang"]');
    const lang = langInput ? langInput.value : 'en';

    // Show skeleton
    card.style.display = 'block';
    grid.replaceChildren();
    const skeletons = document.createDocumentFragment();
    for (let i = 0; i < 6; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'preset-skeleton';
        skeletons.appendChild(skeleton);
    }
    grid.appendChild(skeletons);

    try {
        const [prompts, categories] = await Promise.all([
            API.getPrompts(lang),
            API.getPromptCategories(lang)
        ]);

        allPresets = prompts;
        presetCategories = categories;

        renderCategoryFilter();
        renderPresetTiles();
    } catch (e) {
        ToastManager.error('Failed to load prompt presets');
        grid.replaceChildren();
    }
}

function renderCategoryFilter() {
    const container = document.getElementById('category-filter');
    if (!container) return;

    container.replaceChildren();

    // "All" chip
    const allChip = document.createElement('button');
    allChip.type = 'button';
    allChip.className = 'cat-chip' + (activeCategory === 'all' ? ' active' : '');
    allChip.textContent = 'All';
    allChip.addEventListener('click', () => {
        activeCategory = 'all';
        renderCategoryFilter();
        renderPresetTiles();
    });
    container.appendChild(allChip);

    // Category chips
    presetCategories.forEach(cat => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'cat-chip' + (activeCategory === cat.id ? ' active' : '');
        chip.textContent = cat.name;
        chip.addEventListener('click', () => {
            activeCategory = cat.id;
            renderCategoryFilter();
            renderPresetTiles();
        });
        container.appendChild(chip);
    });
}

function renderPresetTiles() {
    const grid = document.getElementById('prompts-grid');
    if (!grid) return;

    grid.replaceChildren();

    const filtered = activeCategory === 'all'
        ? allPresets
        : allPresets.filter(p => p.category === activeCategory);

    if (filtered.length === 0) {
        const empty = document.createElement('p');
        empty.style.color = 'var(--text-dim)';
        empty.style.gridColumn = '1/-1';
        empty.textContent = 'No presets available';
        grid.appendChild(empty);
        return;
    }

    // Get currently selected preset
    const sel = document.getElementById('prompt_preset');
    const selectedId = sel ? sel.value : '';

    filtered.forEach(p => {
        const tile = document.createElement('div');
        tile.className = 'preset-tile';
        if (p.id === selectedId) tile.classList.add('selected');

        const nameEl = document.createElement('div');
        nameEl.className = 'preset-tile-name';
        nameEl.textContent = p.name || p.id;

        const descEl = document.createElement('div');
        descEl.className = 'preset-tile-desc';
        descEl.textContent = p.description || '';

        tile.append(nameEl, descEl);
        if (p.category) {
            const categoryEl = document.createElement('div');
            categoryEl.className = 'preset-tile-category';
            categoryEl.textContent = p.category;
            tile.appendChild(categoryEl);
        }

        tile.addEventListener('click', () => {
            // Update the select dropdown
            if (sel) {
                sel.value = p.id;
                sel.dispatchEvent(new Event('change'));
            }
            // Update selected state
            document.querySelectorAll('.preset-tile').forEach(t => t.classList.remove('selected'));
            tile.classList.add('selected');
        });

        grid.appendChild(tile);
    });
}

// ── Stop ──────────────────────────────────────────────────────────────────────
async function stopJob() {
    closeSSE();
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

const App = Object.freeze({
    clearLog,
    refreshModels,
    loadPromptPresets,
    stopJob,
    suggestOutputName
});

function bindActionButton(id, handler) {
    const button = document.getElementById(id);
    if (button) button.addEventListener('click', handler);
}

function bindActionButtons() {
    bindActionButton('clear-input-file-btn', () => {
        const input = document.getElementById('input_file');
        const fileName = document.getElementById('file-name');
        if (input) input.value = '';
        if (fileName) fileName.textContent = '—';
        input?.focus();
    });
    bindActionButton('clear-output-file-btn', () => {
        const input = document.getElementById('output_file');
        if (input) { input.value = ''; input.focus(); }
    });
    bindActionButton('suggest-output-btn', App.suggestOutputName);
    bindActionButton('refresh-models-btn', App.refreshModels);
    bindActionButton('stop-btn', App.stopJob);
    bindActionButton('refresh-prompts-btn', App.loadPromptPresets);
    bindActionButton('clear-log-btn', App.clearLog);
}

window.AIRewriterApp = App;
window.addEventListener('beforeunload', closeSSE);

// ── Form submit ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize components
    ToastManager.init();
    UIRenderer.init();
    
    await initI18n();
    bindActionButtons();

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

    // Clear buttons
    document.querySelectorAll('.clear-btn[data-target]').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = document.getElementById(btn.dataset.target);
            if (target) { target.value = ''; target.focus(); }
        });
    });

    // Theme toggle
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') document.body.classList.add('light-mode');
        themeBtn.addEventListener('click', () => {
            document.body.classList.toggle('light-mode');
            localStorage.setItem('theme', document.body.classList.contains('light-mode') ? 'light' : 'dark');
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
    loadPromptPresets();
});

})();
