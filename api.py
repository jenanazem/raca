"""
RACA Legal LLM — FastAPI Inference Server
==========================================
Serves the fine-tuned Llama model with SAE-based hallucination steering.

Usage:
    pip install fastapi uvicorn
    python api.py

Endpoints:
    POST /ask          — Ask a legal question (with steering)
    POST /ask/compare  — Compare plain vs steered responses
    GET  /health       — Health check
"""

import json
import time
import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "./checkpoints/ft_raca/merged"
SAE_PATH       = "./checkpoints/sae_raca/sae_best.pt"
FEATURE_FILE   = "./checkpoints/sae_raca/feature_interpretations.json"
CONFIG_FILE    = "./data/processed/project_config.json"
HOOK_LAYER     = 16
STEER_COEFF    = 0.5
MAX_NEW_TOKENS = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── SAE definition (must match phase4) ────────────────────────────────────────

class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.empty(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


# ── Activation steerer ────────────────────────────────────────────────────────

class ActivationSteerer:
    def __init__(self, model, sae, hal_feature_ids, layer_idx, coeff):
        self.sae = sae
        self.hal_ids = hal_feature_ids
        self.coeff = coeff
        self.hook = None
        self._register(model, layer_idx)

    def _register(self, model, layer_idx):
        target = model.model.layers[layer_idx]
        sae = self.sae
        hal_ids = self.hal_ids
        coeff = self.coeff

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            orig_dtype = h.dtype
            orig_shape = h.shape
            h_flat = h.float().reshape(-1, h.shape[-1])
            with torch.no_grad():
                sae.to(h_flat.device)
                feat = sae.encode(h_flat)
                feat_mod = feat.clone()
                feat_mod[:, hal_ids] *= (1 - coeff)
                correction = (sae.decode(feat_mod) - sae.decode(feat)).reshape(orig_shape).to(orig_dtype)
                h_new = h + correction
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        self.hook = target.register_forward_hook(hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


# ── Load everything at startup ────────────────────────────────────────────────

print("Loading model and SAE...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Load SAE
checkpoint = torch.load(SAE_PATH, map_location="cpu")
d_model = checkpoint["W_enc"].shape[0]
d_sae   = checkpoint["W_enc"].shape[1]
sae = SparseAutoencoder(d_model, d_sae)
sae.load_state_dict(checkpoint)
sae.eval()

# Load hallucination feature IDs
with open(FEATURE_FILE) as f:
    features = json.load(f)
hal_ids = [
    int(fid) for fid, fdata in features.items()
    if fdata.get("is_hallucination_feature") is True
]
print(f"Loaded {len(hal_ids)} hallucination features to suppress: {hal_ids}")

print("Ready.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RACA Legal LLM",
    description="Arabic legal question answering with SAE-based hallucination reduction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    steering_coefficient: Optional[float] = STEER_COEFF


class AnswerResponse(BaseModel):
    question: str
    answer: str
    steered: bool
    latency_ms: float


class CompareResponse(BaseModel):
    question: str
    answer_plain: str
    answer_steered: str
    latency_plain_ms: float
    latency_steered_ms: float


def generate(question: str, max_tokens: int, use_steering: bool, coeff: float) -> tuple[str, float]:
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    steerer = None
    if use_steering and hal_ids:
        steerer = ActivationSteerer(model, sae, hal_ids, HOOK_LAYER, coeff)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - t0) * 1000

    if steerer:
        steerer.remove()

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, latency


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "hal_features_suppressed": len(hal_ids),
        "device": str(next(model.parameters()).device),
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer, latency = generate(req.question, req.max_tokens, use_steering=True, coeff=req.steering_coefficient)
    return AnswerResponse(
        question=req.question,
        answer=answer,
        steered=True,
        latency_ms=round(latency, 1),
    )


@app.post("/ask/compare", response_model=CompareResponse)
def ask_compare(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer_plain,   lat_plain   = generate(req.question, req.max_tokens, use_steering=False, coeff=req.steering_coefficient)
    answer_steered, lat_steered = generate(req.question, req.max_tokens, use_steering=True,  coeff=req.steering_coefficient)
    return CompareResponse(
        question=req.question,
        answer_plain=answer_plain,
        answer_steered=answer_steered,
        latency_plain_ms=round(lat_plain, 1),
        latency_steered_ms=round(lat_steered, 1),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
