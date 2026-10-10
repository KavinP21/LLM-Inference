#!/usr/bin/env python3
"""Minimal native desktop chat UI for the Apple MLX Forge runtime."""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from forge_chat_runtime import CHAT_PROFILES, ChatWorker

ROOT = Path(__file__).resolve().parents[1]
MAX_GENERATION_TOKENS = 32768


class ForgeChat(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Forge LLM")
        self.geometry("760x620")
        self.minsize(620, 480)
        self.configure(background="#f5f5f7")

        self.ready = False
        self.busy = False
        self.loading = False
        self.closing = False
        self.profile = CHAT_PROFILES[0]
        self.conversation: list[dict[str, str]] = []
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
            width=49,
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
        self.max_tokens.set("512")
        self.max_tokens.pack(side="left", padx=(8, 14))
        self.generate_button = ttk.Button(
            controls, text="Generate", command=self._generate, state="disabled"
        )
        self.generate_button.pack(side="left")
        self.clear_button = ttk.Button(
            controls, text="Clear chat", command=self._clear_chat
        )
        self.clear_button.pack(side="left", padx=(8, 0))

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

    def _generate(self) -> None:
        if self.busy or self.loading or self.closing or not self.ready:
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
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
        self._set_controls()
        self._set_status("Generating…", "#8a5a00")
        self.worker.generate(
            [*self.conversation, {"role": "user", "content": prompt}], max_tokens
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
                    prompt, response = value  # type: ignore[misc]
                    self.conversation.extend(
                        [
                            {"role": "user", "content": str(prompt)},
                            {"role": "assistant", "content": str(response)},
                        ]
                    )
                    self._render_conversation()
                    self.prompt.delete("1.0", "end")
                    self.busy = False
                    self._set_controls()
                    self._set_status(self._ready_status(), "#188038")
                elif kind == "error":
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
            lines.append(f"{speaker}\n{message['content'].strip()}")
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
