#!/usr/bin/env python3
"""Minimal native desktop chat UI for the Apple MLX Forge runtime."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from forge_chat_runtime import CHAT_PROFILES, ChatWorker, record_final_reply

ROOT = Path(__file__).resolve().parents[1]
MAX_GENERATION_TOKENS = 32768


class ForgeChat(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Forge LLM")
        self.geometry("800x740")
        self.minsize(700, 660)
        self.configure(background="#f5f5f7")

        self.ready = False
        self.busy = False
        self.loading = False
        self.closing = False
        self.profile = CHAT_PROFILES[0]
        self.conversation: list[dict] = []
        self.attachments: list[dict] = []
        self.last_reasoning = ""
        self.submitted_prompt: str | None = None
        self.context_limit = 0
        self.device_name = "Metal"
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_ui()
        self.worker = ChatWorker(ROOT, self.events)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._drain_events)
        self._start_loading()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=22)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="Forge LLM", font=("SF Pro Display", 24, "bold")).pack(
            anchor="w"
        )
        self.model_label = ttk.Label(
            header,
            text=f"{self.profile.label}  ·  Apple MLX",
            foreground="#666666",
        )
        self.model_label.pack(anchor="w", pady=(2, 0))
        self.artifact_label = ttk.Label(
            header,
            text=f"Model: {self.profile.artifact}",
            foreground="#666666",
        )
        self.artifact_label.pack(anchor="w", pady=(2, 0))
        self.model_picker = ttk.Combobox(
            header,
            values=[profile.label for profile in CHAT_PROFILES],
            state="readonly",
            width=55,
        )
        self.model_picker.current(0)
        self.model_picker.pack(anchor="w", pady=(8, 0))
        self.model_picker.bind("<<ComboboxSelected>>", self._select_model)

        status_row = ttk.Frame(outer)
        status_row.pack(fill="x", pady=(0, 14))
        self.status_dot = tk.Label(status_row, text="●", fg="#8a5a00", bg="#f5f5f7")
        self.status_dot.pack(side="left")
        self.status_label = ttk.Label(status_row, text="Loading model…")
        self.status_label.pack(side="left", padx=(6, 0))
        self.load_button = ttk.Button(
            status_row, text="Load model", command=self._start_loading
        )
        self.load_button.pack(side="right")

        ttk.Label(outer, text="Prompt", font=("SF Pro Text", 11, "bold")).pack(
            anchor="w"
        )
        self.prompt = tk.Text(
            outer,
            height=5,
            wrap="word",
            font=("SF Mono", 12),
            padx=10,
            pady=10,
            relief="flat",
            background="white",
            foreground="#111111",
            insertbackground="#111111",
        )
        self.prompt.pack(fill="x", pady=(6, 12))
        self.prompt.insert("1.0", "Explain how paged KV caching works.")

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 16))
        ttk.Label(controls, text="Max new tokens").pack(side="left")
        self.max_tokens = ttk.Spinbox(
            controls, from_=1, to=MAX_GENERATION_TOKENS, width=8
        )
        self.max_tokens.set("2048")
        self.max_tokens.pack(side="left", padx=(8, 14))
        self.generate_button = ttk.Button(
            controls, text="Generate", command=self._generate, state="disabled"
        )
        self.generate_button.pack(side="left")
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self._stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(
            controls, text="Clear chat", command=self._clear_chat
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        media_controls = ttk.Frame(outer)
        media_controls.pack(fill="x", pady=(0, 8))
        self.thinking = tk.BooleanVar(value=False)
        self.thinking_button = ttk.Checkbutton(
            media_controls, text="Thinking", variable=self.thinking
        )
        self.thinking_button.pack(side="left")
        self.image_button = ttk.Button(
            media_controls, text="Attach image", command=lambda: self._attach("image")
        )
        self.image_button.pack(side="left", padx=(12, 0))
        self.video_button = ttk.Button(
            media_controls,
            text="Attach short video",
            command=lambda: self._attach("video"),
        )
        self.video_button.pack(side="left", padx=(8, 0))
        self.remove_media_button = ttk.Button(
            media_controls, text="Remove attachments", command=self._remove_attachments
        )
        self.remove_media_button.pack(side="left", padx=(8, 0))
        self.attachment_label = ttk.Label(
            outer, text="No attachments", foreground="#666666", wraplength=740
        )
        self.attachment_label.pack(anchor="w", pady=(0, 12))

        ttk.Label(outer, text="Conversation", font=("SF Pro Text", 11, "bold")).pack(
            anchor="w"
        )
        response_frame = ttk.Frame(outer)
        response_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.output = tk.Text(
            response_frame,
            wrap="word",
            state="disabled",
            font=("SF Pro Text", 13),
            padx=12,
            pady=12,
            relief="flat",
            background="white",
            foreground="#111111",
            insertbackground="#111111",
        )
        self.output.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(response_frame, command=self.output.yview)
        scrollbar.pack(side="right", fill="y")
        self.output.configure(yscrollcommand=scrollbar.set)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=text)
        self.status_dot.configure(fg=color)

    def _start_loading(self) -> None:
        if self.loading or self.busy or self.closing:
            return
        self.loading = True
        self.ready = False
        self.context_limit = 0
        self._set_controls()
        self.load_button.configure(state="disabled", text="Loading…")
        self._set_status("Loading model…", "#8a5a00")
        self.worker.load(self.profile)

    def _select_model(self, _event=None) -> None:
        if self.loading or self.busy or self.closing:
            return
        profile = CHAT_PROFILES[self.model_picker.current()]
        if profile.key == self.profile.key:
            return
        self.profile = profile
        self.conversation.clear()
        self.attachments.clear()
        self.last_reasoning = ""
        self.submitted_prompt = None
        self.worker.media_store.clear()
        self.thinking.set(False)
        self.max_tokens.set("2048" if profile.supports_thinking else "512")
        self._update_attachments()
        self._write_output("")
        self.model_label.configure(text=f"{profile.label}  ·  Apple MLX")
        self.artifact_label.configure(text=f"Model: {profile.artifact}")
        self._start_loading()

    def _set_controls(self) -> None:
        idle = not (self.loading or self.busy or self.closing)
        self.model_picker.configure(state="readonly" if idle else "disabled")
        self.load_button.configure(state="normal" if idle else "disabled")
        self.clear_button.configure(state="normal" if idle else "disabled")
        self.max_tokens.configure(state="normal" if idle else "disabled")
        self.generate_button.configure(
            state="normal" if idle and self.ready else "disabled"
        )
        self.stop_button.configure(
            state="normal" if self.busy and not self.closing else "disabled"
        )
        media = idle and self.ready and self.profile.supports_media
        self.image_button.configure(state="normal" if media else "disabled")
        self.video_button.configure(state="normal" if media else "disabled")
        self.remove_media_button.configure(
            state="normal" if media and self.attachments else "disabled"
        )
        self.thinking_button.configure(
            state="normal" if idle and self.profile.supports_thinking else "disabled"
        )

    def _attach(self, kind: str) -> None:
        if (
            self.busy
            or self.loading
            or self.closing
            or not self.ready
            or not self.profile.supports_media
        ):
            return
        limit = 4 if kind == "image" else 1
        history_count = sum(
            part.get("type") == kind
            for message in self.conversation
            if isinstance(message["content"], list)
            for part in message["content"]
        )
        if (
            history_count + sum(part["type"] == kind for part in self.attachments)
            >= limit
        ):
            messagebox.showinfo(
                "Attachment limit",
                f"This chat supports up to {limit} {kind} files. Clear chat to attach more.",
            )
            return
        patterns = (
            "*.png *.jpg *.jpeg *.webp *.bmp"
            if kind == "image"
            else "*.mp4 *.mov *.m4v *.webm"
        )
        source = filedialog.askopenfilename(
            title=f"Choose a local {kind}", filetypes=[(kind.title(), patterns)]
        )
        if not source:
            return
        try:
            self.attachments.append(self.worker.media_store.stage(Path(source), kind))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Attachment error", str(exc))
        self._update_attachments()
        self._set_controls()

    def _update_attachments(self) -> None:
        labels = []
        for part in self.attachments:
            path = part[part["type"]]
            labels.append(
                f"{part['type'].title()}: {self.worker.media_store.names.get(path, Path(path).name)}"
            )
        self.attachment_label.configure(
            text=" · ".join(labels) if labels else "No attachments"
        )

    def _remove_attachments(self) -> None:
        if self.busy or self.loading or self.closing:
            return
        # Pending copies have not entered history; keep older visual turns intact.
        for part in self.attachments:
            self.worker.media_store.discard(part)
        self.attachments.clear()
        self._update_attachments()
        self._set_controls()

    def _stop(self) -> None:
        if self.busy and not self.closing:
            self.worker.stop()
            self.stop_button.configure(state="disabled")
            self._set_status("Stopping…", "#8a5a00")

    def _generate(self) -> None:
        if self.busy or self.loading or self.closing or not self.ready:
            return
        raw_prompt = self.prompt.get("1.0", "end-1c")
        prompt = raw_prompt.strip()
        if not prompt and not self.attachments:
            messagebox.showinfo("Forge LLM", "Enter a prompt first.")
            return
        try:
            max_tokens = int(self.max_tokens.get())
            if not 1 <= max_tokens <= MAX_GENERATION_TOKENS:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid token limit",
                f"Max new tokens must be between 1 and {MAX_GENERATION_TOKENS:,}.",
            )
            return

        self.busy = True
        self.submitted_prompt = raw_prompt
        self._set_controls()
        self._set_status("Generating…", "#8a5a00")
        content = prompt or "Describe the attached media."
        if self.attachments:
            content = [{"type": "text", "text": content}, *self.attachments]
        self.worker.generate(
            [*self.conversation, {"role": "user", "content": content}],
            max_tokens,
            enable_thinking=self.thinking.get() and self.profile.supports_thinking,
        )

    def _drain_events(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "ready":
                    self.profile, self.context_limit, self.device_name = value  # type: ignore[misc]
                    self.ready = True
                    self.loading = False
                    self.load_button.configure(state="normal", text="Reload model")
                    self._set_controls()
                    self._set_status(self._ready_status(), "#188038")
                elif kind == "response":
                    user_message, response = value  # type: ignore[misc]
                    self.last_reasoning = response.reasoning
                    if response.finish_reason == "cancelled":
                        self.submitted_prompt = None
                        self.busy = False
                        self._set_controls()
                        self._set_status(f"Stopped · {self._ready_status()}", "#8a5a00")
                        continue
                    accepted = record_final_reply(
                        self.conversation, user_message, response
                    )
                    if accepted:
                        if self.prompt.get("1.0", "end-1c") == self.submitted_prompt:
                            self.prompt.delete("1.0", "end")
                        self.attachments.clear()
                        self._update_attachments()
                    self._render_conversation()
                    self.submitted_prompt = None
                    self.busy = False
                    self._set_controls()
                    if not response.thinking_complete:
                        self._set_status(
                            "Token limit reached during thinking; no final answer yet.",
                            "#8a5a00",
                        )
                    elif response.truncated:
                        self._set_status(
                            "Token limit reached · answer may be incomplete", "#8a5a00"
                        )
                    elif not accepted:
                        self._set_status(
                            "No final answer returned; increase the token limit or try again.",
                            "#8a5a00",
                        )
                    else:
                        self._set_status(self._ready_status(), "#188038")
                elif kind == "error":
                    self.submitted_prompt = None
                    self.loading = False
                    self.busy = False
                    self.load_button.configure(
                        state="normal",
                        text="Reload model" if self.ready else "Load model",
                    )
                    self._set_controls()
                    if self.ready:
                        self._set_status(f"Error · {self._ready_status()}", "#c5221f")
                    else:
                        self.generate_button.configure(state="disabled")
                        self._set_status("Error", "#c5221f")
                    self._write_output(f"Error: {value}")
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def _write_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def _render_conversation(self) -> None:
        lines = []
        for message in self.conversation:
            speaker = "You" if message["role"] == "user" else "Forge"
            content = message["content"]
            if isinstance(content, list):
                pieces = []
                for part in content:
                    if part["type"] == "text":
                        pieces.append(part["text"])
                    else:
                        path = part[part["type"]]
                        pieces.append(
                            f"[{part['type'].title()}: {self.worker.media_store.names.get(path, Path(path).name)}]"
                        )
                content = "\n".join(pieces)
            lines.append(f"{speaker}\n{content.strip()}")
        if self.last_reasoning:
            lines.append(
                f"Thinking (latest response; excluded from chat history)\n{self.last_reasoning}"
            )
        self._write_output("\n\n".join(lines))

    def _ready_status(self) -> str:
        context = (
            f"{self.context_limit // 1024}K context"
            if self.context_limit
            else "context ready"
        )
        return f"Ready · {self.device_name} · {context}"

    def _clear_chat(self) -> None:
        if self.busy or self.loading or self.closing:
            return
        self.conversation.clear()
        self.attachments.clear()
        self.last_reasoning = ""
        self.submitted_prompt = None
        self.worker.media_store.clear()
        self._update_attachments()
        self._set_controls()
        self._write_output("")
        if self.ready:
            self._set_status(self._ready_status(), "#188038")

    def _close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.worker.close()
        self.destroy()


if __name__ == "__main__":
    # The app bundle sets PYTHONPATH; this fallback keeps direct script launches convenient.
    import sys

    sys.path.insert(0, str(ROOT / "python"))
    ForgeChat().mainloop()
