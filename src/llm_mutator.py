"""
llm_mutator.py — LLM-based sequence mutation engine.

Constructs a structured Chain-of-Thought prompt, queries a frozen causal LM,
and parses the output for a valid peptide sequence.
"""
from __future__ import annotations

import re
import warnings
from typing import Optional

from src.config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_TOP_P, MOTIF, SEQ_LEN

# ─── Prompt Template ─────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Context: You are an expert protein engineer specializing in constrained peptide design.
Task: Given a parent peptide sequence and a desired shift on a charge-hydropathy grid, \
propose a single mutated peptide that moves in the requested direction while remaining \
biophysically plausible.

Parent Sequence: {parent}
Target Shift: {target_shift}

Constraints:
1. MUST output exactly {seq_len} amino-acid residues using the standard 20-letter alphabet (ACDEFGHIKLMNPQRSTVWY).
2. MUST contain the contiguous motif "{motif}" exactly once.
3. MUST NOT use any amino acid more than 4 times.
4. MUST contain at least one aromatic residue from {{F, W, Y}}.
5. DO NOT include spaces, punctuation, or any extra characters in the final sequence.

Provide your reasoning step-by-step, then output the final answer in the exact format below:

FINAL_SEQUENCE: <sequence>"""


def build_prompt(parent: str, target_shift: str) -> str:
    return _PROMPT_TEMPLATE.format(
        parent=parent,
        target_shift=target_shift,
        seq_len=SEQ_LEN,
        motif=MOTIF,
    )


def parse_output(text: str) -> Optional[str]:
    """Extract FINAL_SEQUENCE from LLM output text."""
    match = re.search(r"FINAL[_\s]SEQUENCE\s*[:=]\s*([A-Za-z]{5,30})", text, re.IGNORECASE)
    if match:
        return match.group(1).upper().strip()
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"[A-Z]{15}", line):
            return line
    return None


# ─── LLM Mutator ─────────────────────────────────────────────────────────────

class LLMMutator:
    """Frozen causal LM used as a proposal operator for MAP-Elites.

    Supports any HuggingFace-compatible model (Qwen, Mistral, Llama, etc.).
    The model weights are never updated.
    """

    def __init__(self, model_name_or_path: str, device: str = "auto",
                 load_in_8bit: bool = False, load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name_or_path
        self.device = device

        print(f"[LLMMutator] Loading: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(trust_remote_code=True, device_map=device)
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            kwargs["load_in_4bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        self.model.eval()
        print("[LLMMutator] Ready.")

    def mutate(self, parent: str, target_shift: str, n_attempts: int = 3) -> Optional[str]:
        import torch

        prompt = build_prompt(parent, target_shift)
        for attempt in range(n_attempts):
            try:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        top_p=LLM_TOP_P,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                candidate = parse_output(text)
                if candidate:
                    return candidate
            except Exception as e:
                warnings.warn(f"[LLMMutator] Attempt {attempt+1}: {e}")
        return None


class MockMutator:
    """Deterministic mock mutator for testing (no GPU required)."""

    def __init__(self, model_name_or_path: str = "mock"):
        self.model_name = model_name_or_path
        print("[MockMutator] Initialized.")

    def mutate(self, parent: str, target_shift: str, n_attempts: int = 3) -> Optional[str]:
        import random
        seq = list(parent)
        rgd_pos = parent.find("RGD")
        protected = set(range(rgd_pos, rgd_pos + 3)) if rgd_pos >= 0 else set()
        mutable = [i for i in range(len(seq)) if i not in protected]

        if mutable:
            pos = random.choice(mutable)
            seq[pos] = random.choice(list("ACDEFGHIKLMNPQRSTVWY"))

        if not any(aa in "FWY" for aa in seq) and mutable:
            seq[mutable[0]] = "Y"

        return "".join(seq)
