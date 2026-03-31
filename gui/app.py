"""
Tkinter GUI for AI Book Rewriter.
Supports dark theme, i18n (ru/en/zh), prompt presets.
"""
import os
import queue
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import sv_ttk

from core.config import FINAL_SUFFIX, LOCAL_MODEL_NAME as DEFAULT_MODEL
from core.services import RewriteService, ModelProvider, RewriteParams, RewriteStatus
from core.services.rewrite_service import get_rewrite_service
from core.services.model_provider import get_model_provider
from core.prompts import get_preset_names
from core.services.prompt_service import get_prompt_service
from i18n import tr, set_language, get_supported_languages, get_output_languages

logger = logging.getLogger(__name__)
log_queue: queue.Queue = queue.Queue()


class QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(self.format(record))


class BookRewriterApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        self._current_lang = "ru"
        # Service instances
        self._rewrite_service: RewriteService = get_rewrite_service()
        self._model_provider: ModelProvider = get_model_provider()
        self._setup_logging()
        self._build_ui()
        self._refresh_models()
        master.after(100, self._poll_log)

    # ── Logging ──────────────────────────────────────────────────────────────
    def _setup_logging(self):
        handler = QueueHandler(log_queue)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logging.getLogger().addHandler(handler)

    # ── UI Build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.master.title(tr("app_title"))
        self.master.geometry("1000x780")
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Top bar: language switcher
        top_bar = ttk.Frame(self.master, padding="6 4")
        top_bar.pack(fill=tk.X)
        ttk.Label(top_bar, text=tr("interface_language") + ":").pack(side=tk.LEFT, padx=(0, 6))
        self.lang_iface_var = tk.StringVar(value=self._current_lang)
        lang_cb = ttk.Combobox(
            top_bar, textvariable=self.lang_iface_var,
            values=list(get_supported_languages().keys()), state="readonly", width=6
        )
        lang_cb.pack(side=tk.LEFT)
        lang_cb.bind("<<ComboboxSelected>>", self._on_lang_change)

        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Settings
        self._sf = ttk.LabelFrame(main_frame, text=tr("settings"))
        self._sf.pack(pady=(0, 8), fill=tk.X)

        self._widgets: dict = {}  # key -> widget reference for i18n updates
        self._build_settings(self._sf)

        # Controls
        ctrl = ttk.Frame(main_frame)
        ctrl.pack(fill=tk.X, pady=4)
        self.start_btn = ttk.Button(ctrl, text=tr("start"), command=self._start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn  = ttk.Button(ctrl, text=tr("stop"),  command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value=tr("status_ready"))
        ttk.Label(ctrl, textvariable=self.status_var, anchor=tk.E).pack(side=tk.RIGHT, padx=4)
        self.progress = ttk.Progressbar(ctrl, mode="determinate", length=320)
        self.progress.pack(side=tk.RIGHT, padx=4)

        # Log
        log_frame = ttk.LabelFrame(main_frame, text=tr("log"))
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        self.log_area = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, height=14)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_settings(self, parent: ttk.LabelFrame):
        parent.grid_columnconfigure(1, weight=1)

        def row(r, label_key, widget_factory):
            lbl = ttk.Label(parent, text=tr(label_key))
            lbl.grid(row=r, column=0, sticky=tk.W, padx=6, pady=3)
            self._widgets[label_key + "_lbl"] = lbl
            w = widget_factory(parent)
            w.grid(row=r, column=1, sticky=tk.EW, padx=6, pady=3)
            return w

        # Input file
        inp_frame = ttk.Frame(parent)
        self.input_var = tk.StringVar()
        ttk.Entry(inp_frame, textvariable=self.input_var, width=55).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._widgets["browse_in"] = ttk.Button(inp_frame, text=tr("browse"), command=self._browse_input)
        self._widgets["browse_in"].pack(side=tk.LEFT, padx=4)
        lbl_in = ttk.Label(parent, text=tr("input_file"))
        lbl_in.grid(row=0, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["input_file_lbl"] = lbl_in
        inp_frame.grid(row=0, column=1, sticky=tk.EW, padx=6, pady=3)
        
        # Output file
        out_frame = ttk.Frame(parent)
        self.output_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_var, width=55).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._widgets["browse_out"] = ttk.Button(out_frame, text=tr("browse"), command=self._browse_output)
        self._widgets["browse_out"].pack(side=tk.LEFT, padx=4)
        lbl_out = ttk.Label(parent, text=tr("output_file"))
        lbl_out.grid(row=1, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["output_file_lbl"] = lbl_out
        out_frame.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=3)
        
        # Language
        self.lang_var = tk.StringVar(value="Русский")
        lang_combo = ttk.Combobox(parent, textvariable=self.lang_var,
                                  values=get_output_languages(), state="readonly", width=22)
        lbl_lg = ttk.Label(parent, text=tr("language"))
        lbl_lg.grid(row=2, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["language_lbl"] = lbl_lg
        lang_combo.grid(row=2, column=1, sticky=tk.W, padx=6, pady=3)
        
        # Model
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(parent, textvariable=self.model_var, state="readonly", width=42)
        lbl_m = ttk.Label(parent, text=tr("model"))
        lbl_m.grid(row=3, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["model_lbl"] = lbl_m
        self.model_combo.grid(row=3, column=1, sticky=tk.EW, padx=6, pady=3)
        
        # Prompt preset
        self.preset_var = tk.StringVar(value="literary")
        self.preset_combo = ttk.Combobox(parent, textvariable=self.preset_var, state="readonly", width=30)
        # Prompt preview/description - created BEFORE _refresh_preset_combo() to avoid AttributeError
        self.preset_preview = ttk.Label(parent, text="", wraplength=400, foreground="#8b90b0")
        self._refresh_preset_combo()
        lbl_pr = ttk.Label(parent, text=tr("prompt_preset"))
        lbl_pr.grid(row=4, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["prompt_preset_lbl"] = lbl_pr
        self.preset_combo.grid(row=4, column=1, sticky=tk.W, padx=6, pady=3)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)
        self.preset_preview.grid(row=5, column=1, sticky=tk.W, padx=6, pady=(0, 6))
        
        # Style
        lbl_st = ttk.Label(parent, text=tr("style"))
        lbl_st.grid(row=5, column=0, sticky=tk.NW, padx=6, pady=3)
        self._widgets["style_lbl"] = lbl_st
        self.style_text = tk.Text(parent, height=3, width=50)
        self.style_text.grid(row=5, column=1, columnspan=2, sticky=tk.EW, padx=6, pady=3)
        self.style_text.insert("1.0", tr("default_style"))
        
        # Goal
        lbl_go = ttk.Label(parent, text=tr("goal"))
        lbl_go.grid(row=6, column=0, sticky=tk.NW, padx=6, pady=3)
        self._widgets["goal_lbl"] = lbl_go
        self.goal_text = tk.Text(parent, height=4, width=50)
        self.goal_text.grid(row=6, column=1, columnspan=2, sticky=tk.EW, padx=6, pady=3)
        self.goal_text.insert("1.0", tr("default_goal"))
        
        # Resume
        self.resume_var = tk.BooleanVar(value=True)
        lbl_re = ttk.Label(parent, text=tr("resume"))
        lbl_re.grid(row=7, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["resume_lbl"] = lbl_re
        ttk.Checkbutton(parent, variable=self.resume_var).grid(row=7, column=1, sticky=tk.W, padx=6, pady=3)

        # Parallel mode
        self.parallel_var = tk.BooleanVar(value=False)
        lbl_pa = ttk.Label(parent, text=tr("parallel_mode"))
        lbl_pa.grid(row=8, column=0, sticky=tk.W, padx=6, pady=3)
        self._widgets["parallel_mode_lbl"] = lbl_pa
        ttk.Checkbutton(parent, variable=self.parallel_var).grid(row=8, column=1, sticky=tk.W, padx=6, pady=3)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _refresh_models(self):
        models = self._model_provider.get_available_models()
        self.model_combo["values"] = models
        if models:
            self.model_var.set(models[0] if DEFAULT_MODEL not in models else DEFAULT_MODEL)

    def _refresh_preset_combo(self):
        """Refresh preset combobox with localized names from PromptService."""
        prompt_service = get_prompt_service()
        prompts = prompt_service.get_all_prompts(self._current_lang)
        self._preset_labels = {p.id: p.name for p in prompts}
        self.preset_combo["values"] = list(self._preset_labels.values())
        # Store mapping for reverse lookup
        self._preset_ids = {p.name: p.id for p in prompts}
        # Set current selection
        current_id = self.preset_var.get()
        if current_id in self._preset_labels:
            self.preset_combo.set(self._preset_labels[current_id])
        self._update_preset_preview()

    def _on_preset_change(self, _event=None):
        """Handle preset selection change."""
        self._update_preset_preview()

    def _update_preset_preview(self):
        """Update the preset preview label with description."""
        selected_label = self.preset_combo.get()
        prompt_id = self._preset_ids.get(selected_label, "literary")
        prompt_service = get_prompt_service()
        prompt = prompt_service.get_prompt_by_id(prompt_id, self._current_lang)
        if prompt:
            self.preset_preview.configure(text=prompt.description or "")
            # Update preset_var with the ID
            self.preset_var.set(prompt_id)

    def _browse_input(self):
        f = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if f:
            self.input_var.set(f)
            # Auto-fill output file if empty
            if not self.output_var.get().strip():
                stem = Path(f).stem
                output_file = str(Path(f).parent / (stem + FINAL_SUFFIX))
                self.output_var.set(output_file)
    
    def _browse_output(self):
        f = filedialog.asksaveasfilename(
            filetypes=[("Text files", "*.txt")],
            defaultextension=".txt",
            initialfile=self.output_var.get().strip() or None,
        )
        if f:
            self.output_var.set(f)
    
    # ── Language change ───────────────────────────────────────────────────────
    def _on_lang_change(self, _event=None):
        lang = self.lang_iface_var.get()
        self._current_lang = lang
        set_language(lang)
        self._refresh_preset_combo()
        # Update visible labels
        label_keys = [
            "input_file", "output_file", "language", "model", "prompt_preset",
            "style", "goal", "resume", "parallel_mode",
        ]
        for k in label_keys:
            w = self._widgets.get(k + "_lbl")
            if w:
                w.configure(text=tr(k))
        browse_btn = self._widgets.get("browse_in")
        if browse_btn:
            browse_btn.configure(text=tr("browse"))
        browse_out_btn = self._widgets.get("browse_out")
        if browse_out_btn:
            browse_out_btn.configure(text=tr("browse"))
        self.start_btn.configure(text=tr("start"))
        self.stop_btn.configure(text=tr("stop"))
        self.master.title(tr("app_title"))
        self._sf.configure(text=tr("settings"))
    
    # ── Start / Stop ──────────────────────────────────────────────────────────
    def _start(self):
        input_file = self.input_var.get().strip()
        if not input_file:
            messagebox.showerror(tr("app_title"), tr("error_fill_fields"))
            return
        if not os.path.exists(input_file):
            messagebox.showerror(tr("app_title"), tr("error_file_not_found", path=input_file))
            return
    
        # Use user-specified output file or auto-generate
        output_file = self.output_var.get().strip()
        if not output_file:
            stem = Path(input_file).stem
            output_file = str(Path(input_file).parent / (stem + FINAL_SUFFIX))
    
        params = RewriteParams(
            input_file=input_file,
            output_file=output_file,
            language=self.lang_var.get(),
            style=self.style_text.get("1.0", tk.END).strip(),
            goal=self.goal_text.get("1.0", tk.END).strip(),
            model=self.model_var.get(),
            resume=self.resume_var.get(),
            parallel=self.parallel_var.get(),
            save_interval=1,
            prompt_preset=self.preset_var.get(),
        )

        self.start_btn["state"] = tk.DISABLED
        self.stop_btn["state"] = tk.NORMAL
        self.status_var.set(tr("status_starting"))
        self.progress["value"] = 0

        started = self._rewrite_service.start_rewrite(
            params=params,
            progress_callback=self._progress_safe,
        )
        
        if not started:
            messagebox.showerror(tr("app_title"), "Failed to start rewrite")
            self.start_btn["state"] = tk.NORMAL
            self.stop_btn["state"] = tk.DISABLED

    def _stop(self):
        if self._rewrite_service.is_running():
            self.status_var.set(tr("status_stopping"))
            self._rewrite_service.stop_rewrite()
            self.stop_btn["state"] = tk.DISABLED

    def _progress_safe(self, current: int, total: int):
        self.master.after(0, self._update_progress, current, total)

    def _update_progress(self, current: int, total: int):
        if total > 0:
            self.progress["maximum"] = total
            self.progress["value"]   = current
            pct = current / total * 100
            self.status_var.set(tr("status_running", current=current, total=total, pct=pct))
        else:
            self.progress["value"] = 0

    # ── Log polling ───────────────────────────────────────────────────────────
    def _poll_log(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_area.configure(state=tk.NORMAL)
                self.log_area.insert(tk.END, msg + "\n")
                self.log_area.configure(state=tk.DISABLED)
                self.log_area.yview(tk.END)
        except queue.Empty:
            pass

        # Check service status
        status = self._rewrite_service.get_status()
        if status.status in (RewriteStatus.COMPLETED, RewriteStatus.FAILED, RewriteStatus.STOPPED):
            if status.status == RewriteStatus.STOPPED:
                final = tr("status_stopped")
            elif status.status == RewriteStatus.FAILED:
                final = tr("status_done") + f" ({status.error_message or 'error'})"
            else:
                final = tr("status_done")
            self.status_var.set(final)
            self.start_btn["state"] = tk.NORMAL
            self.stop_btn["state"] = tk.DISABLED

        self.master.after(100, self._poll_log)

    def _on_closing(self):
        if self._rewrite_service.is_running():
            if messagebox.askyesno(tr("confirm_exit_title"), tr("confirm_exit_msg")):
                self._stop()
                self.master.after(300, self.master.destroy)
            else:
                self.master.destroy()
        else:
            self.master.destroy()


def run_gui():
    """Run the GUI application."""
    root = tk.Tk()
    sv_ttk.set_theme("dark")
    BookRewriterApp(root)
    root.mainloop()


# NOTE: Direct execution via `python gui/app.py` is no longer supported.
# Use one of the following instead:
#   python -m app.gui
#   python main.py --gui
