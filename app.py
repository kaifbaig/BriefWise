from pathlib import Path
import logging
import re

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, T5ForConditionalGeneration


# Build paths relative to this file, so the app works from the project folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "briefwise_final"
INDEX_FILE = BASE_DIR / "index.html"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BriefWise",
    description="Dialogue summarization using fine-tuned T5-small",
    version="1.0.0",
)

# Select a device: NVIDIA GPU, Apple GPU, or CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Verify model weights availability
weights_file = MODEL_DIR / "model.safetensors"
if not weights_file.is_file():
    logger.error(
        f"Model weights file not found at {weights_file}.\n"
        "BriefWise requires the trained model weights (~242 MB).\n"
        "Please ensure 'model.safetensors' is placed inside './briefwise_final/' before launching."
    )

# Load the saved model and tokenizer once when the app starts.
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
model = model.to(DEVICE)
model.eval()


class DialogueInput(BaseModel):
    dialogue: str = Field(min_length=1, max_length=20000)


def clean_data(text: str) -> str:
    """Apply the same basic text cleaning used in the notebook."""
    text = re.sub(r"<[^>]+>", " ", text)  # Remove simple HTML tags
    text = re.sub(r"\s+", " ", text)      # Normalize whitespace
    return text.strip()                    # Preserve original capitalization


def summarize_dialogue(dialogue: str) -> str:
    """Generate a summary using the saved fine-tuned BriefWise model."""
    dialogue = clean_data(dialogue)

    if not dialogue:
        raise HTTPException(
            status_code=400,
            detail="Please enter a dialogue to summarize.",
        )

    # Match the input format used during training: task prefix + dialogue.
    input_text = "summarize: " + dialogue
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    # Inference does not require gradient calculations.
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            max_length=None,  # Use max_new_tokens without the default max_length warning.
            num_beams=4,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


@app.get("/")
def home():
    """Serve the existing HTML frontend."""
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=500,
            detail="index.html was not found in the project folder.",
        )
    return FileResponse(INDEX_FILE)


@app.post("/summarize/")
def summarize(request: DialogueInput):
    """Accept a dialogue and return its generated summary as JSON."""
    if not request.dialogue.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a dialogue to summarize.",
        )

    try:
        summary = summarize_dialogue(request.dialogue)
        return {"summary": summary}
    except HTTPException:
        raise
    except Exception:
        # Keep technical details in the server console, not in the browser response.
        logger.exception("Summary generation failed")
        raise HTTPException(
            status_code=500,
            detail="Could not generate a summary. Please try again.",
        )


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host=host, port=port)
