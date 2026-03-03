"""
ELI5 inference server.

Loads the merged LoRA model once at startup and serves two endpoints:
  POST /generate         — blocking, returns full response
  POST /generate/stream  — SSE, streams tokens as they're produced
  GET  /health           — liveness check

Usage:
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

Then test with:
    curl -X POST http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Why is the sky blue?"}'
"""

import threading
import torch
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = "models/eli5-merged"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_NEW_TOKENS = 300
SYSTEM_PROMPT = (
    "You are an expert at explaining complex topics in a simple, "
    "easy-to-understand way. You will explain as if I'm a 5-10 year old."
)

# Global model state — loaded once at startup
_model = None
_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer

    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run: python scripts/merge_adapter.py"
        )

    print(f"loading model from {MODEL_PATH} onto {DEVICE}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    _model.eval()
    print("model ready.")

    yield  # server runs here

    # cleanup (optional, process exits anyway)
    del _model, _tokenizer


app = FastAPI(title="ELI5 Inference Server", lifespan=lifespan)


# ── request/response schemas ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = MAX_NEW_TOKENS


class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int


# ── shared: build input ids from prompt ──────────────────────────────────────

def _build_inputs(prompt: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt")
    return inputs["input_ids"].to(DEVICE)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(503, "model not loaded")

    input_ids = _build_inputs(req.prompt)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # decode only the newly generated tokens (strip the prompt)
    new_ids = output_ids[0][input_ids.shape[1]:]
    response_text = _tokenizer.decode(new_ids, skip_special_tokens=True)

    return GenerateResponse(
        response=response_text,
        tokens_generated=len(new_ids),
    )


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    """
    Streams tokens via Server-Sent Events.

    How it works:
      - TextIteratorStreamer sits between the model and the HTTP response.
      - model.generate() runs in a background thread, putting tokens into the streamer queue.
      - The SSE generator reads from that queue and yields events to the client.
    """
    if _model is None:
        raise HTTPException(503, "model not loaded")

    input_ids = _build_inputs(req.prompt)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=False,
        use_cache=True,
        streamer=streamer,
    )

    # generation blocks — run it in a thread so we can stream concurrently
    thread = threading.Thread(target=_model.generate, kwargs=generate_kwargs)
    thread.start()

    def event_generator():
        for token_text in streamer:
            # SSE format: each event is "data: <text>\n\n"
            yield f"data: {token_text}\n\n"
        yield "data: [DONE]\n\n"
        thread.join()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
