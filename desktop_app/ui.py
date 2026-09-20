"""Native tkinter user interface for the DukeOTR local desktop application."""

from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from desktop_app.core.app_core import DukeOTRApplicationCore
from desktop_app.core.builder_adapter import plan_builder_request
from desktop_app.core.models import AppMode, AppSettings, ChatMessage, Conversation, ProviderStatus
from desktop_app.core.modes import ModeSubmission
from desktop_app.core.provider import GenerationCancelled, ModelProviderError
from desktop_app.core.storage import LocalStorageError


_CODE_FENCE = re.compile(r"```(?:[A-Za-z0-9_+.-]+)?\n?(.*?)```", re.DOTALL)


class DukeOTRDesktop(tk.Tk):
    """Responsive native UI; network/model work is always moved off tkinter's event loop."""

    def __init__(self, core: DukeOTRApplicationCore | None = None) -> None:
        super().__init__()
        self.core = core or DukeOTRApplicationCore()
        self.title("DukeOTR · Local AI Workspace")
        self.geometry("1280x800")
        self.minsize(940, 620)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._cancel_event: threading.Event | None = None
        self._generation_token = 0
        self._stream_buffer = ""
        self._stream_mode = AppMode.CHAT
        self._copy_buttons: list[ttk.Button] = []
        self._mode_drafts: dict[AppMode, dict[str, str]] = {mode: {} for mode in AppMode}
        self.current_mode = AppMode.CHAT
        self.current_conversation: Conversation | None = None

        self.model_var = tk.StringVar(value=self.core.settings.default_model)
        self.status_var = tk.StringVar(value="Checking local Ollama…")
        self.model_identity_var = tk.StringVar(value=self._model_identity(self.model_var.get()))
        self.mode_title_var = tk.StringVar(value="Chat")
        self.builder_route_var = tk.StringVar(value="Builder mode prepares an interactive first pass; it does not run the curated quality trace.")

        self._apply_theme(self.core.settings.theme)
        self._build_layout()
        self._populate_conversations()
        if self.core.conversations:
            self._select_conversation(self.core.conversations[0].id)
        else:
            self._new_conversation()
        self._show_storage_warnings()
        self.after(40, self._drain_events)
        self._refresh_connection()

    # ---------- construction and theme ----------

    def _apply_theme(self, preference: str) -> None:
        # Tk has no cross-platform system-theme API. "system" intentionally uses the readable
        # light native palette rather than pretending to detect a setting it cannot reliably read.
        dark = preference == "dark"
        palette = (
            {
                "bg": "#111827",
                "surface": "#172033",
                "surface_2": "#202d44",
                "text": "#edf2f7",
                "muted": "#a9b6cb",
                "accent": "#60a5fa",
                "accent_dark": "#1d4ed8",
                "border": "#31415e",
                "user": "#193450",
                "assistant": "#19283d",
                "code": "#0b1220",
                "warning": "#fbbf24",
                "error": "#fca5a5",
            }
            if dark
            else {
                "bg": "#f4f7fb",
                "surface": "#ffffff",
                "surface_2": "#eaf1fb",
                "text": "#172033",
                "muted": "#61708a",
                "accent": "#2563eb",
                "accent_dark": "#1d4ed8",
                "border": "#cfdaea",
                "user": "#e5f0ff",
                "assistant": "#f5f8fc",
                "code": "#111827",
                "warning": "#9a6700",
                "error": "#b42318",
            }
        )
        self._palette = palette
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(background=palette["bg"])
        style.configure("TFrame", background=palette["bg"])
        style.configure("Surface.TFrame", background=palette["surface"])
        style.configure("Sidebar.TFrame", background=palette["surface"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["text"], font=("Segoe UI", 10))
        style.configure("Surface.TLabel", background=palette["surface"], foreground=palette["text"])
        style.configure("Muted.TLabel", background=palette["surface"], foreground=palette["muted"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=palette["surface"], foreground=palette["text"], font=("Segoe UI Semibold", 16))
        style.configure("Mode.TLabel", background=palette["surface"], foreground=palette["accent"], font=("Segoe UI Semibold", 11))
        style.configure("Status.TLabel", background=palette["surface"], foreground=palette["muted"], font=("Segoe UI", 9))
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=palette["accent"], foreground="#ffffff", borderwidth=0)
        style.map("Accent.TButton", background=[("active", palette["accent_dark"]), ("disabled", palette["border"])])
        style.configure("Mode.TButton", anchor="w", background=palette["surface"], foreground=palette["text"], borderwidth=0)
        style.map("Mode.TButton", background=[("active", palette["surface_2"]), ("pressed", palette["surface_2"])])
        style.configure("Danger.TButton", foreground=palette["error"])
        style.configure("TEntry", fieldbackground=palette["surface"], foreground=palette["text"], bordercolor=palette["border"])
        style.configure("TCombobox", fieldbackground=palette["surface"], foreground=palette["text"], background=palette["surface"])
        style.configure("TSeparator", background=palette["border"])

    def _build_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(12, 14))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(3, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="DukeOTR", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, text="LOCAL AI WORKSPACE", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 16))
        ttk.Button(sidebar, text="＋  New conversation", style="Accent.TButton", command=self._new_conversation).grid(
            row=2, column=0, sticky="ew", pady=(0, 12)
        )

        conversation_frame = ttk.Frame(sidebar, style="Surface.TFrame")
        conversation_frame.grid(row=3, column=0, sticky="nsew")
        conversation_frame.grid_rowconfigure(1, weight=1)
        conversation_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(conversation_frame, text="CONVERSATIONS", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=(3, 5))
        self.conversation_list = tk.Listbox(
            conversation_frame,
            exportselection=False,
            activestyle="none",
            relief="flat",
            highlightthickness=0,
            background=self._palette["surface"],
            foreground=self._palette["text"],
            selectbackground=self._palette["surface_2"],
            selectforeground=self._palette["text"],
            font=("Segoe UI", 9),
        )
        self.conversation_list.grid(row=1, column=0, sticky="nsew")
        conversation_scroll = ttk.Scrollbar(conversation_frame, orient="vertical", command=self.conversation_list.yview)
        conversation_scroll.grid(row=1, column=1, sticky="ns")
        self.conversation_list.configure(yscrollcommand=conversation_scroll.set)
        self.conversation_list.bind("<<ListboxSelect>>", self._on_conversation_selected)
        ttk.Button(sidebar, text="Delete selected", style="Danger.TButton", command=self._delete_selected).grid(
            row=4, column=0, sticky="ew", pady=(8, 15)
        )

        ttk.Separator(sidebar).grid(row=5, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(sidebar, text="WORKSPACE MODE", style="Muted.TLabel").grid(row=6, column=0, sticky="w", pady=(0, 4))
        self._mode_buttons: dict[AppMode, ttk.Button] = {}
        for index, mode in enumerate(AppMode, start=7):
            button = ttk.Button(sidebar, text=mode.label, style="Mode.TButton", command=lambda selected=mode: self._set_mode(selected))
            button.grid(row=index, column=0, sticky="ew", pady=1)
            self._mode_buttons[mode] = button

        main = ttk.Frame(self, padding=(18, 14))
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(main, style="Surface.TFrame", padding=(14, 10))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        ttk.Label(header, textvariable=self.mode_title_var, style="Mode.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel", wraplength=430).grid(row=1, column=0, sticky="w", pady=(2, 0))
        controls = ttk.Frame(header, style="Surface.TFrame")
        controls.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(controls, textvariable=self.model_identity_var, style="Muted.TLabel").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, width=20, state="normal")
        self.model_combo.grid(row=0, column=1, sticky="e")
        self.model_combo.bind("<<ComboboxSelected>>", self._model_selected)
        self.model_combo.bind("<FocusOut>", self._model_typed)
        ttk.Button(controls, text="↻", width=3, command=self._refresh_connection).grid(row=0, column=2, padx=(7, 0))
        ttk.Button(controls, text="Settings", command=self._open_settings).grid(row=0, column=3, padx=(7, 0))

        content = ttk.Frame(main, style="Surface.TFrame", padding=1)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        self.transcript = ScrolledText(
            content,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=18,
            pady=15,
            font=("Segoe UI", 10),
            background=self._palette["surface"],
            foreground=self._palette["text"],
            insertbackground=self._palette["text"],
        )
        self.transcript.grid(row=0, column=0, sticky="nsew")
        self.transcript.configure(state="disabled")
        self._configure_transcript_tags()

        composer = ttk.Frame(main, style="Surface.TFrame", padding=(14, 11))
        composer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        composer.grid_columnconfigure(0, weight=1)
        self.mode_fields = ttk.Frame(composer, style="Surface.TFrame")
        self.mode_fields.grid(row=0, column=0, sticky="ew")
        self.mode_fields.grid_columnconfigure(1, weight=1)
        self._build_mode_fields()
        self.prompt_input = tk.Text(
            composer,
            height=4,
            wrap="word",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=10,
            pady=8,
            font=("Segoe UI", 10),
            background=self._palette["surface"],
            foreground=self._palette["text"],
            insertbackground=self._palette["text"],
        )
        self.prompt_input.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.prompt_input.insert("1.0", "Ask about Roblox, Luau, or your project…")
        self.prompt_input.bind("<FocusIn>", self._clear_input_hint)
        action_bar = ttk.Frame(composer, style="Surface.TFrame")
        action_bar.grid(row=2, column=0, sticky="ew")
        action_bar.grid_columnconfigure(0, weight=1)
        ttk.Label(action_bar, text="Ctrl+Enter to send · all history stays on this computer", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.stop_button = ttk.Button(action_bar, text="Stop", command=self._stop_generation, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(8, 0))
        self.send_button = ttk.Button(action_bar, text="Send  ›", style="Accent.TButton", command=self._send)
        self.send_button.grid(row=0, column=2, padx=(8, 0))

        self.bind_all("<Control-Return>", self._send_from_shortcut)
        self.bind_all("<Control-n>", lambda event: self._new_from_shortcut(event))
        self.bind_all("<Escape>", lambda event: self._stop_from_shortcut(event))

    def _configure_transcript_tags(self) -> None:
        self.transcript.tag_configure("role_user", foreground=self._palette["accent"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("role_assistant", foreground=self._palette["text"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("role_interrupted", foreground=self._palette["warning"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("message", foreground=self._palette["text"], spacing3=10)
        self.transcript.tag_configure("code", foreground="#e5edf8", background=self._palette["code"], font=("Cascadia Mono", 9), lmargin1=12, lmargin2=12, rmargin=12, spacing1=5, spacing3=5)
        self.transcript.tag_configure("meta", foreground=self._palette["muted"], font=("Segoe UI", 8))
        self.transcript.tag_configure("error", foreground=self._palette["error"], font=("Segoe UI", 10))

    # ---------- conversation and transcript ----------

    def _populate_conversations(self) -> None:
        selected_id = self.current_conversation.id if self.current_conversation else None
        self.conversation_list.delete(0, "end")
        self._conversation_ids: list[str] = []
        for conversation in self.core.conversations:
            self.conversation_list.insert("end", conversation.title)
            self._conversation_ids.append(conversation.id)
        if selected_id and selected_id in self._conversation_ids:
            position = self._conversation_ids.index(selected_id)
            self.conversation_list.selection_set(position)

    def _new_conversation(self) -> None:
        if self._generation_active():
            self._set_status("Finish or stop the current generation before starting another conversation.", "warning")
            return
        try:
            conversation = self.core.new_conversation(self.current_mode)
        except LocalStorageError as exc:
            self._set_status(str(exc), "error")
            return
        self.current_conversation = conversation
        self._populate_conversations()
        self._select_conversation(conversation.id)
        self.prompt_input.delete("1.0", "end")
        self.prompt_input.focus_set()

    def _new_from_shortcut(self, event: tk.Event[Any]) -> str:
        self._new_conversation()
        return "break"

    def _on_conversation_selected(self, _event: tk.Event[Any]) -> None:
        selection = self.conversation_list.curselection()
        if not selection or self._generation_active():
            return
        self._select_conversation(self._conversation_ids[selection[0]])

    def _select_conversation(self, conversation_id: str) -> None:
        conversation = self.core.get_conversation(conversation_id)
        if conversation is None:
            return
        self.current_conversation = conversation
        self._set_mode(conversation.mode, persist=False)
        self._populate_conversations()
        if conversation_id in self._conversation_ids:
            position = self._conversation_ids.index(conversation_id)
            self.conversation_list.selection_clear(0, "end")
            self.conversation_list.selection_set(position)
            self.conversation_list.see(position)
        self._render_transcript()

    def _delete_selected(self) -> None:
        if self._generation_active() or self.current_conversation is None:
            return
        if not messagebox.askyesno("Delete conversation", f"Delete ‘{self.current_conversation.title}’ from local history?"):
            return
        selected = self.current_conversation.id
        try:
            self.core.delete_conversation(selected)
        except LocalStorageError as exc:
            self._set_status(str(exc), "error")
            return
        self.current_conversation = None
        self._populate_conversations()
        if self.core.conversations:
            self._select_conversation(self.core.conversations[0].id)
        else:
            self._new_conversation()

    def _render_transcript(self) -> None:
        for button in self._copy_buttons:
            button.destroy()
        self._copy_buttons.clear()
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        if self.current_conversation is None or not self.current_conversation.messages:
            self.transcript.insert("end", "Welcome to DukeOTR\n", "role_assistant")
            self.transcript.insert(
                "end",
                "Choose a mode, confirm the locally installed Ollama model, and start a conversation. "
                "DukeOTR is the application; the model selector always shows the actual model tag.\n",
                "message",
            )
        else:
            for message in self.current_conversation.messages:
                self._insert_message(message)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _insert_message(self, message: ChatMessage) -> None:
        if message.role == "user":
            title, tag = "You", "role_user"
        elif message.interrupted:
            title, tag = "DukeOTR · interrupted", "role_interrupted"
        else:
            title, tag = f"DukeOTR · {message.model or 'local model'}", "role_assistant"
        self.transcript.insert("end", title + "\n", tag)
        self._insert_rich_text(message.content)
        if message.mode:
            suffix = f"{message.mode.label} mode"
            if message.interrupted:
                suffix += " · partial response saved locally"
            self.transcript.insert("end", suffix + "\n\n", "meta")
        else:
            self.transcript.insert("end", "\n\n")

    def _insert_rich_text(self, content: str) -> None:
        position = 0
        for match in _CODE_FENCE.finditer(content):
            before = content[position : match.start()]
            if before:
                self.transcript.insert("end", before, "message")
            code = match.group(1).strip("\n")
            if code:
                self.transcript.insert("end", code + "\n", "code")
                button = ttk.Button(self.transcript, text="Copy code", command=lambda value=code: self._copy_code(value))
                self._copy_buttons.append(button)
                self.transcript.window_create("end", window=button, padx=4, pady=3)
                self.transcript.insert("end", "\n")
            position = match.end()
        remainder = content[position:]
        if remainder:
            self.transcript.insert("end", remainder, "message")

    def _copy_code(self, value: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(value)
        self.update_idletasks()
        self._set_status("Code copied to the clipboard.", "connected")

    # ---------- modes and mode fields ----------

    def _set_mode(self, mode: AppMode, *, persist: bool = True) -> None:
        if self._generation_active():
            self._set_status("Finish or stop generation before changing modes.", "warning")
            return
        self._save_current_field_draft()
        self.current_mode = mode
        self.mode_title_var.set(mode.label)
        self._build_mode_fields()
        if persist and self.current_conversation is not None:
            self.current_conversation.mode = mode
            self.current_conversation.touch()
            try:
                self.core.save_conversation(self.current_conversation)
            except LocalStorageError as exc:
                self._set_status(str(exc), "error")
                return
            self._populate_conversations()
        for candidate, button in self._mode_buttons.items():
            button.configure(text=("●  " if candidate == mode else "   ") + candidate.label)

    def _save_current_field_draft(self) -> None:
        if not hasattr(self, "mode_fields"):
            return
        draft: dict[str, str] = {}
        for name in ("action_var", "language_var", "error_var", "expected_var", "actual_var"):
            variable = getattr(self, name, None)
            if variable is not None:
                draft[name] = variable.get()
        code_input = getattr(self, "code_input", None)
        if code_input is not None:
            draft["code"] = code_input.get("1.0", "end-1c")
        self._mode_drafts[self.current_mode] = draft

    def _clear_mode_widgets(self) -> None:
        for child in self.mode_fields.winfo_children():
            child.destroy()
        for name in ("action_var", "language_var", "error_var", "expected_var", "actual_var", "code_input"):
            if hasattr(self, name):
                delattr(self, name)

    def _build_mode_fields(self) -> None:
        self._clear_mode_widgets()
        draft = self._mode_drafts[self.current_mode]
        if self.current_mode == AppMode.CHAT:
            ttk.Label(self.mode_fields, text="General local conversation", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            return
        if self.current_mode == AppMode.BUILDER:
            ttk.Label(self.mode_fields, textvariable=self.builder_route_var, style="Muted.TLabel", wraplength=900).grid(
                row=0, column=0, sticky="w"
            )
            return

        self.language_var = tk.StringVar(value=draft.get("language_var", "Luau"))
        ttk.Label(self.mode_fields, text="Language", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        language = ttk.Combobox(self.mode_fields, textvariable=self.language_var, values=("Luau", "Lua", "Python", "TypeScript", "Other"), width=12)
        language.grid(row=0, column=1, sticky="w", padx=(0, 12))
        if self.current_mode == AppMode.CODE:
            self.action_var = tk.StringVar(value=draft.get("action_var", "Generate"))
            ttk.Label(self.mode_fields, text="Action", style="Muted.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
            ttk.Combobox(
                self.mode_fields,
                textvariable=self.action_var,
                values=("Generate", "Explain", "Improve", "Fix", "Refactor"),
                width=12,
                state="readonly",
            ).grid(row=0, column=3, sticky="w")

        if self.current_mode == AppMode.DEBUG:
            self.error_var = tk.StringVar(value=draft.get("error_var", ""))
            self.expected_var = tk.StringVar(value=draft.get("expected_var", ""))
            self.actual_var = tk.StringVar(value=draft.get("actual_var", ""))
            ttk.Label(self.mode_fields, text="Error / symptom", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(7, 0))
            ttk.Entry(self.mode_fields, textvariable=self.error_var).grid(row=1, column=1, columnspan=3, sticky="ew", pady=(7, 0))
            ttk.Label(self.mode_fields, text="Expected", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(5, 0))
            ttk.Entry(self.mode_fields, textvariable=self.expected_var).grid(row=2, column=1, sticky="ew", pady=(5, 0))
            ttk.Label(self.mode_fields, text="Actual", style="Muted.TLabel").grid(row=2, column=2, sticky="w", padx=(9, 6), pady=(5, 0))
            ttk.Entry(self.mode_fields, textvariable=self.actual_var).grid(row=2, column=3, sticky="ew", pady=(5, 0))

        label = "Code to review" if self.current_mode in {AppMode.REVIEW, AppMode.SECURITY} else "Optional code"
        row = 3 if self.current_mode == AppMode.DEBUG else 1
        ttk.Label(self.mode_fields, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=(7, 3))
        self.code_input = ScrolledText(
            self.mode_fields,
            height=5,
            wrap="none",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            padx=8,
            pady=5,
            font=("Cascadia Mono", 9),
            background=self._palette["surface"],
            foreground=self._palette["text"],
            insertbackground=self._palette["text"],
        )
        self.code_input.grid(row=row + 1, column=0, columnspan=4, sticky="ew")
        self.code_input.insert("1.0", draft.get("code", ""))

    def _submission(self) -> ModeSubmission:
        prompt = self.prompt_input.get("1.0", "end-1c").strip()
        if prompt == "Ask about Roblox, Luau, or your project…":
            prompt = ""
        code = self.code_input.get("1.0", "end-1c") if hasattr(self, "code_input") else ""
        language = self.language_var.get() if hasattr(self, "language_var") else "Luau"
        action = self.action_var.get() if hasattr(self, "action_var") else "Generate"
        error = self.error_var.get() if hasattr(self, "error_var") else ""
        expected = self.expected_var.get() if hasattr(self, "expected_var") else ""
        actual = self.actual_var.get() if hasattr(self, "actual_var") else ""
        route_summary = ""
        if self.current_mode == AppMode.BUILDER:
            planning_input = "\n".join(value for value in (prompt, code) if value.strip())
            context = plan_builder_request(planning_input)
            route_summary = context.summary()
            self.builder_route_var.set(route_summary)
        return ModeSubmission(
            mode=self.current_mode,
            prompt=prompt,
            code=code,
            language=language,
            action=action,
            error=error,
            expected=expected,
            actual=actual,
            builder_route_summary=route_summary,
        )

    # ---------- sending, streaming, cancellation ----------

    def _send_from_shortcut(self, _event: tk.Event[Any]) -> str:
        self._send()
        return "break"

    def _send(self) -> None:
        if self._generation_active():
            return
        if self.current_conversation is None:
            self._new_conversation()
        if self.current_conversation is None:
            return
        # A user can type a model tag directly into the editable selector without moving focus.
        # Commit that local choice before constructing the request so UI and provider stay aligned.
        if not self._save_selected_model(silent=True):
            self._set_status("Choose a local model before sending.", "error")
            return
        try:
            submission = self._submission()
            prepared = self.core.prepare_generation(self.current_conversation, submission)
        except ValueError as exc:
            self._set_status(str(exc), "error")
            return

        try:
            self.core.record_user_message(self.current_conversation, prepared.user_message)
        except LocalStorageError as exc:
            self._set_status(str(exc), "error")
            return
        self._populate_conversations()
        self._render_transcript()
        self._append_stream_start()
        self._generation_token += 1
        token = self._generation_token
        self._stream_mode = submission.mode
        self._stream_buffer = ""
        self._cancel_event = threading.Event()
        self.send_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_status(f"Generating locally with {prepared.request.model}…", "working")
        worker = threading.Thread(
            target=self._generation_worker,
            args=(token, prepared.request, self._cancel_event),
            daemon=True,
            name="DukeOTR-Ollama-Stream",
        )
        worker.start()
        self.prompt_input.delete("1.0", "end")

    def _generation_worker(self, token: int, request: Any, cancel_event: threading.Event) -> None:
        parts: list[str] = []
        active_model = request.model
        try:
            for chunk in self.core.stream_generation(request, cancel_event):
                if chunk.content:
                    parts.append(chunk.content)
                    self._events.put(("generation_chunk", {"token": token, "content": chunk.content}))
                if chunk.model:
                    active_model = chunk.model
            self._events.put(("generation_done", {"token": token, "content": "".join(parts), "model": active_model}))
        except GenerationCancelled as exc:
            self._events.put(
                ("generation_cancelled", {"token": token, "content": "".join(parts), "model": active_model, "message": str(exc)})
            )
        except ModelProviderError as exc:
            self._events.put(("generation_error", {"token": token, "message": str(exc)}))
        except Exception as exc:  # Defensive boundary: UI stays usable if a provider implementation has a bug.
            self._events.put(("generation_error", {"token": token, "message": f"Unexpected local application error: {exc}"}))

    def _append_stream_start(self) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"DukeOTR · {self.model_var.get()}\n", "role_assistant")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _append_stream_chunk(self, content: str) -> None:
        self._stream_buffer += content
        self.transcript.configure(state="normal")
        self.transcript.insert("end", content, "message")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _stop_generation(self) -> None:
        if not self._generation_active() or self._cancel_event is None:
            return
        self._cancel_event.set()
        self.stop_button.configure(state="disabled")
        self._set_status("Stopping local generation…", "working")

    def _stop_from_shortcut(self, _event: tk.Event[Any]) -> str:
        self._stop_generation()
        return "break"

    def _generation_active(self) -> bool:
        return self._cancel_event is not None

    def _finish_generation(self) -> None:
        self._cancel_event = None
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    # ---------- background provider status ----------

    def _refresh_connection(self) -> None:
        if self._generation_active():
            self._set_status("Generation is active; connection refresh waits for it to finish.", "warning")
            return
        self._set_status("Checking local Ollama connection…", "working")
        threading.Thread(target=self._status_worker, daemon=True, name="DukeOTR-Ollama-Status").start()

    def _status_worker(self) -> None:
        status = self.core.refresh_provider_status()
        self._events.put(("provider_status", {"status": status}))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, data = self._events.get_nowait()
                if kind == "provider_status":
                    self._apply_provider_status(data["status"])
                elif data.get("token") == self._generation_token:
                    self._handle_generation_event(kind, data)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(40, self._drain_events)

    def _handle_generation_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "generation_chunk":
            self._append_stream_chunk(data["content"])
            return
        if self.current_conversation is None:
            self._finish_generation()
            return
        if kind == "generation_done":
            content = data["content"].strip()
            if content:
                try:
                    self.core.record_assistant_message(
                        self.current_conversation,
                        content,
                        model=data["model"],
                        mode=self._stream_mode,
                    )
                except LocalStorageError as exc:
                    # Keep the already-rendered streamed text visible for copying in this window,
                    # but make clear it was not added to durable local history.
                    self._set_status(f"Generation completed, but the answer could not be saved: {exc}", "error")
                else:
                    self._populate_conversations()
                    self._render_transcript()
                    self._set_status(f"Local generation complete · {data['model']}", "connected")
            else:
                self._append_transcript_error("Ollama completed without visible assistant text. No empty answer was saved.")
                self._set_status("Generation completed without text.", "error")
            self._finish_generation()
            return
        if kind == "generation_cancelled":
            content = data["content"].strip()
            storage_error: LocalStorageError | None = None
            if content:
                try:
                    self.core.record_assistant_message(
                        self.current_conversation,
                        content,
                        model=data["model"],
                        mode=self._stream_mode,
                        interrupted=True,
                    )
                except LocalStorageError as exc:
                    storage_error = exc
                    # The partial stream remains visible until the user leaves this conversation,
                    # but it is not represented as saved history.
                else:
                    self._populate_conversations()
                    self._render_transcript()
            else:
                self._append_transcript_error("Generation stopped before any assistant text was received. The user message remains in local history.")
            if storage_error is not None:
                self._set_status(f"Partial response could not be saved: {storage_error}", "error")
            else:
                self._set_status("Generation stopped. Any received partial response was saved locally and marked interrupted.", "warning")
            self._finish_generation()
            return
        if kind == "generation_error":
            self._append_transcript_error(data["message"])
            self._set_status("Generation failed; see the transcript for an actionable message.", "error")
            self._finish_generation()

    def _append_transcript_error(self, text: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", "\n\nGeneration error\n", "role_interrupted")
        self.transcript.insert("end", text + "\n", "error")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _apply_provider_status(self, status: ProviderStatus) -> None:
        # A delayed startup refresh must not overwrite the visible "Generating"/"Stopping"
        # state with stale connection text while a request is in flight.
        generation_active = self._generation_active()
        if status.connected:
            models = [model.name for model in status.models]
            self.model_combo.configure(values=models)
            selected = self.model_var.get().strip() or self.core.settings.default_model
            self.model_var.set(selected)
            self.model_identity_var.set(self._model_identity(selected))
            if generation_active:
                return
            self._set_status(status.message, "connected")
            if selected not in models:
                self._set_status(
                    f"{status.message} · Selected model {selected!r} is not listed. Choose an installed model before sending.",
                    "warning",
                )
        else:
            self.model_combo.configure(values=())
            if not generation_active:
                self._set_status(status.message, "error")

    # ---------- model and settings ----------

    def _model_identity(self, model: str) -> str:
        return "Qwen3-4B (qwen3:4b)" if model.strip() == "qwen3:4b" else f"Model: {model.strip() or 'not selected'}"

    def _model_selected(self, _event: tk.Event[Any]) -> None:
        self._save_selected_model()

    def _model_typed(self, _event: tk.Event[Any]) -> None:
        self._save_selected_model(silent=True)

    def _save_selected_model(self, *, silent: bool = False) -> bool:
        try:
            self.core.select_model(self.model_var.get())
        except ValueError as exc:
            if not silent:
                self._set_status(str(exc), "error")
            return False
        self.model_identity_var.set(self._model_identity(self.core.settings.default_model))
        if not silent:
            self._set_status(f"Selected local model: {self.core.settings.default_model}", "connected")
        return True

    def _open_settings(self) -> None:
        if self._generation_active():
            self._set_status("Finish or stop generation before changing settings.", "warning")
            return
        dialog = tk.Toplevel(self)
        dialog.title("DukeOTR settings")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(background=self._palette["surface"])
        dialog.grab_set()
        body = ttk.Frame(dialog, style="Surface.TFrame", padding=18)
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)
        current = self.core.settings
        url_var = tk.StringVar(value=current.ollama_url)
        model_var = tk.StringVar(value=current.default_model)
        temperature_var = tk.StringVar(value=str(current.temperature))
        context_var = tk.StringVar(value=str(current.num_ctx))
        predict_var = tk.StringVar(value=str(current.num_predict))
        theme_var = tk.StringVar(value=current.theme)
        streaming_var = tk.BooleanVar(value=current.response_streaming)

        fields = [
            ("Ollama HTTP URL", url_var),
            ("Default model tag", model_var),
            ("Temperature (0–2)", temperature_var),
            ("Context window", context_var),
            ("Maximum response tokens", predict_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(body, text=label, style="Surface.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
            ttk.Entry(body, textvariable=variable, width=34).grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Label(body, text="Theme", style="Surface.TLabel").grid(row=5, column=0, sticky="w", padx=(0, 14), pady=5)
        ttk.Combobox(body, textvariable=theme_var, values=("system", "light", "dark"), state="readonly", width=18).grid(row=5, column=1, sticky="w", pady=5)
        ttk.Checkbutton(body, text="Stream response text when Ollama supports it", variable=streaming_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(7, 3))
        ttk.Label(
            body,
            text=(
                f"History and settings remain local: {self.core.settings_store.directory}\n"
                "The app uses Ollama's HTTP API and never runs a terminal command or downloads models automatically."
            ),
            style="Muted.TLabel",
            wraplength=470,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 12))
        controls = ttk.Frame(body, style="Surface.TFrame")
        controls.grid(row=8, column=0, columnspan=2, sticky="e")
        ttk.Button(controls, text="Cancel", command=dialog.destroy).grid(row=0, column=0, padx=(0, 8))

        def save() -> None:
            try:
                settings = AppSettings(
                    ollama_url=url_var.get(),
                    default_model=model_var.get(),
                    temperature=float(temperature_var.get()),
                    num_ctx=int(context_var.get()),
                    num_predict=int(predict_var.get()),
                    theme=theme_var.get(),
                    response_streaming=streaming_var.get(),
                    data_directory=str(self.core.settings_store.directory),
                )
                self.core.update_settings(settings)
            except ValueError as exc:
                messagebox.showerror("Invalid settings", str(exc), parent=dialog)
                return
            except LocalStorageError as exc:
                messagebox.showerror("Settings not saved", str(exc), parent=dialog)
                return
            self.model_var.set(settings.default_model)
            self.model_identity_var.set(self._model_identity(settings.default_model))
            self._apply_theme(settings.theme)
            # Update widget colors that Tk Text/Listbox do not receive through ttk styles.
            self._refresh_widget_palette()
            dialog.destroy()
            self._set_status("Settings saved locally. Checking configured Ollama endpoint…", "working")
            self._refresh_connection()

        ttk.Button(controls, text="Save settings", style="Accent.TButton", command=save).grid(row=0, column=1)
        dialog.bind("<Return>", lambda _event: save())

    def _refresh_widget_palette(self) -> None:
        self.conversation_list.configure(
            background=self._palette["surface"],
            foreground=self._palette["text"],
            selectbackground=self._palette["surface_2"],
            selectforeground=self._palette["text"],
        )
        for widget in (self.transcript, self.prompt_input):
            widget.configure(background=self._palette["surface"], foreground=self._palette["text"], insertbackground=self._palette["text"])
        if hasattr(self, "code_input"):
            self.code_input.configure(background=self._palette["surface"], foreground=self._palette["text"], insertbackground=self._palette["text"])
        self._configure_transcript_tags()
        self._render_transcript()

    # ---------- UI feedback and lifecycle ----------

    def _set_status(self, text: str, state: str) -> None:
        prefix = {"connected": "● ", "working": "◌ ", "warning": "▲ ", "error": "● "}.get(state, "")
        self.status_var.set(prefix + text)
        color = {
            "connected": self._palette["accent"],
            "working": self._palette["muted"],
            "warning": self._palette["warning"],
            "error": self._palette["error"],
        }.get(state, self._palette["muted"])
        ttk.Style(self).configure("Status.TLabel", foreground=color)

    def _clear_input_hint(self, _event: tk.Event[Any]) -> None:
        value = self.prompt_input.get("1.0", "end-1c")
        if value == "Ask about Roblox, Luau, or your project…":
            self.prompt_input.delete("1.0", "end")

    def _show_storage_warnings(self) -> None:
        warnings = self.core.storage_warnings
        if warnings:
            self._set_status(" ".join(warnings), "warning")

    def _on_close(self) -> None:
        if self._generation_active() and self._cancel_event is not None:
            self._cancel_event.set()
        self.destroy()


def run_desktop_app() -> None:
    """Entry point kept separate from importable UI code for Windows packaging."""

    app = DukeOTRDesktop()
    app.mainloop()
