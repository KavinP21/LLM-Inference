#!/usr/bin/env python3
"""Minimal native desktop chat UI for the Apple MLX Forge runtime."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "qwen2.5-0.5b.engine"
MODEL_NAME = "Qwen2.5-0.5B-Instruct"
PREFILL_CHUNK_SIZE = 512
KV_CACHE_BYTES = 512 << 20
MAX_GENERATION_TOKENS = 32768


class ForgeChat(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Forge LLM")
        self.geometry("760x620")
        self.minsize(620, 480)
        self.configure(background="#f5f5f7")

        self.engine = None
        self.tokenizer = None
        self.busy = False
        self.loading = False
        self.conversation: list[dict[str, str]] = []
        self.context_limit = 0
        self.device_name = "Metal"
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_ui()
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
        ttk.Label(
            header,
            text=f"{MODEL_NAME}  ·  Apple MLX",
            foreground="#666666",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            header,
            text=f"Model: {MODEL_PATH.name}",
            foreground="#666666",
        ).pack(anchor="w", pady=(2, 0))

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
        if self.loading or self.busy:
            return
        self.loading = True
        if self.engine is not None:
            self.engine.close()
        self.engine = None
        self.tokenizer = None
        self.generate_button.configure(state="disabled")
        self.load_button.configure(state="disabled", text="Loading…")
        self._set_status("Loading model…", "#8a5a00")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            from forge_llm import create_engine
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            engine = create_engine(
                MODEL_PATH,
                backend="mlx",
                max_num_sequences=1,
                max_model_length=None,
                kv_cache_bytes=KV_CACHE_BYTES,
                prefill_chunk_size=PREFILL_CHUNK_SIZE,
            )
            self.events.put(("ready", (engine, tokenizer)))
        except Exception as exc:  # surfaced in the UI so terminal use is unnecessary
            self.events.put(("error", str(exc)))

    def _generate(self) -> None:
        if self.busy or self.engine is None or self.tokenizer is None:
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
        self.generate_button.configure(state="disabled")
        self._set_status("Generating…", "#8a5a00")
        threading.Thread(
            target=self._run_generation, args=(prompt, max_tokens), daemon=True
        ).start()

    def _run_generation(self, prompt: str, max_tokens: int) -> None:
        try:
            messages = [*self.conversation, {"role": "user", "content": prompt}]
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer(rendered, return_tensors=None).input_ids
            available_tokens = self.context_limit - len(tokens)
            if available_tokens <= 0:
                raise ValueError(
                    f"This conversation is too long for the {self.context_limit:,}-token context. "
                    "Press Clear chat to start a new conversation."
                )
            output_tokens = self.engine.generate(
                tokens,
                max_new_tokens=min(max_tokens, available_tokens),
                eos_token_ids=[self.tokenizer.eos_token_id],
            )
            response = self.tokenizer.decode(
                output_tokens, skip_special_tokens=True
            ).strip()
            self.events.put(("response", (prompt, response)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "ready":
                    self.engine, self.tokenizer = value  # type: ignore[misc]
                    self.context_limit = int(self.engine.max_model_length)
                    self.device_name = str(
                        self.engine.build_info().get("device", "Metal")
                    )
                    self.loading = False
                    self.load_button.configure(state="normal", text="Reload model")
                    self.generate_button.configure(state="normal")
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
                    self.generate_button.configure(state="normal")
                    self._set_status(self._ready_status(), "#188038")
                elif kind == "error":
                    self.loading = False
                    self.busy = False
                    self.load_button.configure(
                        state="normal",
                        text="Reload model"
                        if self.engine is not None
                        else "Load model",
                    )
                    if self.engine is not None:
                        self.generate_button.configure(state="normal")
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
        if self.busy:
            return
        self.conversation.clear()
        self._write_output("")
        self._set_status(self._ready_status(), "#188038")

    def _close(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.destroy()


if __name__ == "__main__":
    # The app bundle sets PYTHONPATH; this fallback keeps direct script launches convenient.
    import sys

    sys.path.insert(0, str(ROOT / "python"))
    ForgeChat().mainloop()
