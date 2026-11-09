"""Load Gemma 4 E2B with the Resonance adapter and generate answers.

Used by the app and by evaluation. Works on a GPU (optionally in 4-bit) or on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from typing import Iterator

from .prompts import build_messages, render_prompt, text_tokenizer

BASE_MODEL = "unsloth/gemma-4-E2B-it"
ADAPTER = "vtaua/resonance-gemma4-physics"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0  # 0 means greedy decoding
    top_p: float = 0.95
    # Kept at 1.0: a strong repetition penalty (the old app used 1.3) punishes the
    # symbols LaTeX must repeat, such as $, \frac and subscripts, and garbles equations.
    repetition_penalty: float = 1.0


@dataclass
class Generation:
    text: str
    new_tokens: int
    truncated: bool


def _model_class():
    # Gemma 4 is a multimodal model; the adapter's weight names include the
    # `language_model` prefix of the conditional-generation class, so load that class.
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("transformers is too old for Gemma 4")


class Tutor:
    def __init__(self, base_model: str = BASE_MODEL, adapter: str | None = ADAPTER,
                 device: str | None = None, load_in_4bit: bool = False):
        import torch
        from transformers import AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        kwargs = {"dtype": torch.bfloat16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = device
        self.tokenizer = text_tokenizer(AutoTokenizer.from_pretrained(adapter or base_model))
        model = _model_class().from_pretrained(base_model, **kwargs)
        if adapter:
            from peft import PeftModel

            # Load the adapter weights on the CPU and move the whole model afterwards: on a
            # ZeroGPU Space there is no GPU at start-up, and PEFT would otherwise load the
            # weights straight onto "cuda".
            model = PeftModel.from_pretrained(model, adapter, torch_device="cpu")
        if not load_in_4bit:
            model = model.to(device)
        model.eval()
        self.model = model
        self.adapter = adapter
        self.stop_ids = self._stop_ids()

    def _stop_ids(self) -> list[int]:
        tok = self.tokenizer
        ids = {tok.eos_token_id}
        end_turn = tok.convert_tokens_to_ids("<turn|>")
        if isinstance(end_turn, int) and end_turn != tok.unk_token_id:
            ids.add(end_turn)
        return sorted(i for i in ids if i is not None)

    def _inputs(self, question: str, history: list[dict] | None):
        prompt = render_prompt(self.tokenizer, build_messages(question, history))
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def _gen_kwargs(self, cfg: GenerationConfig) -> dict:
        kw = {
            "max_new_tokens": cfg.max_new_tokens,
            "eos_token_id": self.stop_ids,
            "pad_token_id": self.tokenizer.pad_token_id or self.stop_ids[0],
            "repetition_penalty": cfg.repetition_penalty,
        }
        if cfg.temperature > 0:
            kw.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            kw.update(do_sample=False)
        return kw

    def generate(self, question: str, history: list[dict] | None = None,
                 cfg: GenerationConfig | None = None, use_adapter: bool = True) -> Generation:
        import contextlib

        import torch

        cfg = cfg or GenerationConfig()
        inputs = self._inputs(question, history)
        ctx = contextlib.nullcontext()
        if not use_adapter and self.adapter:
            ctx = self.model.disable_adapter()
        with torch.no_grad(), ctx:
            out = self.model.generate(**inputs, **self._gen_kwargs(cfg))
        new = out[0][inputs["input_ids"].shape[1]:].tolist()
        stopped = bool(new) and new[-1] in self.stop_ids
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        return Generation(text=text, new_tokens=len(new),
                          truncated=not stopped and len(new) >= cfg.max_new_tokens)

    def stream(self, question: str, history: list[dict] | None = None,
               cfg: GenerationConfig | None = None) -> Iterator[str]:
        """Yield the answer as it grows."""
        from transformers import TextIteratorStreamer

        cfg = cfg or GenerationConfig(temperature=0.3)
        inputs = self._inputs(question, history)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate,
                        kwargs={**inputs, **self._gen_kwargs(cfg), "streamer": streamer})
        thread.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text
        thread.join()
