"""
BriefWise — Streamlit Dialogue Summarization Application
=========================================================
An undergraduate NLP learning project for abstractive dialogue summarization
using a fine-tuned T5-small model trained on the SAMSum corpus.

Supports dual loading modes:
1. Local Development: Loads from `./briefwise_final/` when local weights exist.
2. Cloud Deployment: Loads from private Hugging Face repo `kaifbaig24/briefwise-t5-small`
   using an authenticated read token from Streamlit secrets or `HF_TOKEN` env var.
"""

from pathlib import Path
import logging
import os
import re
import streamlit as st
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
HF_REPO_ID = "kaifbaig24/briefwise-t5-small"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "briefwise_final"
MAX_CHAR_LIMIT = 20000

# Sample dialogues for quick one-click testing
SAMPLE_CONVERSATIONS = {
    "Lunch Meeting Plan": (
        "Hannah: Hey, are you free for lunch today around 1 PM?\n"
        "Amanda: Yes! I was thinking of trying that new Italian place down on 5th Street.\n"
        "Hannah: Luigi's? Sounds great. Should I reserve a table?\n"
        "Amanda: Larry already booked one for four people, so we're good to go!\n"
        "Hannah: Perfect, see you there at 1."
    ),
    "Project Deadline Update": (
        "Sarah: Did you manage to finalize the quarterly analytics report?\n"
        "Tom: Almost done with the charts. I noticed a small discrepancy in the October numbers.\n"
        "Sarah: Good catch. Let's fix that before sending it to the client.\n"
        "Tom: I'll rerun the data pipeline and email the revised draft by 4 PM.\n"
        "Sarah: Thanks Tom, appreciate the quick turnaround!"
    ),
    "Travel & Flight Coordination": (
        "David: Our flight to Chicago is delayed by two hours due to weather.\n"
        "Elena: Oh no, will we still make the connecting flight to Boston?\n"
        "David: The airline customer service rep rebooked us on the 8:30 PM direct flight instead.\n"
        "Elena: That's actually much better, no layover. Let's meet at gate B12."
    )
}

# -----------------------------------------------------------------------------
# Helper Functions: Secrets Retrieval & Text Normalization
# -----------------------------------------------------------------------------

def get_hf_token() -> str | None:
    """
    Safely retrieve Hugging Face read token from Streamlit secrets or environment
    variables without printing, logging, or exposing the token value.
    """
    # 1. Check Streamlit secrets (.streamlit/secrets.toml or Cloud Secrets)
    try:
        if hasattr(st, "secrets"):
            if "HF_TOKEN" in st.secrets:
                token = str(st.secrets["HF_TOKEN"]).strip()
                if token:
                    return token
            if "HUGGINGFACE_HUB_TOKEN" in st.secrets:
                token = str(st.secrets["HUGGINGFACE_HUB_TOKEN"]).strip()
                if token:
                    return token
    except Exception:
        pass

    # 2. Check environment variables
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    return None


def clean_data(text: str) -> str:
    """
    Apply identical text normalization used in training and the FastAPI app:
    - Strips simple HTML tags
    - Collapses consecutive whitespace characters into a single space
    - Preserves casing for proper named-entity recognition
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------------------------------------------------------
# Model & Tokenizer Resource Caching
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """
    Load the fine-tuned T5-small model and tokenizer once per application lifecycle.
    Prefers local `./briefwise_final/` weights; falls back to Hugging Face Hub.
    """
    # Device selection (CUDA GPU -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_label = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        device_label = "CPU"

    # Mode 1: Local development directory
    local_weights = LOCAL_MODEL_DIR / "model.safetensors"
    local_config = LOCAL_MODEL_DIR / "config.json"

    if LOCAL_MODEL_DIR.is_dir() and local_weights.is_file() and local_config.is_file():
        source_desc = f"Local Artifacts (`./briefwise_final`)"
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
        model = T5ForConditionalGeneration.from_pretrained(str(LOCAL_MODEL_DIR))
        model = model.to(device)
        model.eval()
        return model, tokenizer, device, device_label, source_desc

    # Mode 2: Cloud deployment via private Hugging Face Hub repository
    source_desc = f"Hugging Face Hub (`{HF_REPO_ID}`)"
    token = get_hf_token()

    if not token:
        raise PermissionError(
            f"The model weights are hosted in the private repository '{HF_REPO_ID}', "
            "but no Hugging Face read token was found.\n\n"
            "**How to configure your token:**\n"
            "- **On Streamlit Community Cloud:** Go to App Settings > Secrets, and add:\n"
            '  ```toml\n  HF_TOKEN = "hf_your_read_token_here"\n  ```\n'
            "- **Locally:** Create a `.streamlit/secrets.toml` file (see `.streamlit/secrets.toml.example`) "
            "or set the `HF_TOKEN` environment variable."
        )

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, token=token)
    model = T5ForConditionalGeneration.from_pretrained(HF_REPO_ID, token=token)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device, device_label, source_desc


def summarize_conversation(dialogue: str, model, tokenizer, device) -> str:
    """
    Generate an abstractive dialogue summary using beam search.
    Preserves verified generation settings: max 512 input tokens, 4 beams, max 100 new tokens.
    """
    cleaned = clean_data(dialogue)
    if not cleaned:
        raise ValueError("Please enter a dialogue to summarize.")

    # Match input format used during fine-tuning
    input_text = "summarize: " + cleaned
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            max_length=None,
            num_beams=4,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# Streamlit User Interface
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="BriefWise — Dialogue Summarizer",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Sidebar: Model and project information
    with st.sidebar:
        st.title("BriefWise")
        st.caption("Dialogue Summarization with T5-small")
        st.markdown("---")

        st.markdown("### Model Details")
        st.markdown(
            "- **Architecture:** Encoder-Decoder Transformer (T5-small)\n"
            "- **Parameters:** ~60 Million\n"
            "- **Dataset:** SAMSum Messenger Corpus\n"
            "- **Tuning:** Optuna Bayesian Optimization (10 Trials)\n"
            "- **Winning Epoch:** Epoch 3 (Checkpoint 3000)"
        )

        st.markdown("### Independent Benchmark")
        st.markdown(
            "- **ROUGE-1:** `46.88`\n"
            "- **ROUGE-2:** `22.51`\n"
            "- **ROUGE-L:** `38.45`\n"
            "*(Measured over 812 strictly non-overlapping held-out dialogues)*"
        )
        st.markdown("---")

        # Load model and display runtime details
        model_loaded = False
        try:
            with st.spinner("Initializing BriefWise model..."):
                model, tokenizer, device, device_label, source_desc = load_model_and_tokenizer()
            model_loaded = True
            st.success(f"**Model Ready**\n\n- Device: `{device_label}`\n- Source: {source_desc}")
        except PermissionError as pe:
            st.error(f"**Authentication Required**\n\n{str(pe)}")
        except Exception as e:
            logger.exception("Model loading failed")
            st.error(f"**Model Load Error:**\n\n{str(e)}")

        st.markdown("---")
        st.caption("BriefWise Final-Year Project · Built with Streamlit, PyTorch & Transformers")

    # Main Area: Header
    st.title("BriefWise 💬")
    st.subheader("Transform multi-turn conversations into concise, readable summaries.")
    st.markdown(
        "Paste a messenger-style conversation below or select a sample dialogue to see how "
        "the fine-tuned model synthesizes key decisions and actions."
    )

    # Sample dialogue loader
    st.markdown("#### Sample Conversations")
    cols = st.columns(len(SAMPLE_CONVERSATIONS))
    selected_sample = None
    for idx, (label, sample_text) in enumerate(SAMPLE_CONVERSATIONS.items()):
        if cols[idx].button(label, use_container_width=True):
            st.session_state["dialogue_input"] = sample_text

    # Dialogue input area
    dialogue_value = st.session_state.get("dialogue_input", "")
    dialogue_input = st.text_area(
        label="Conversation Dialogue",
        value=dialogue_value,
        height=220,
        placeholder=(
            "Hannah: Hey, are you free for lunch today around 1 PM?\n"
            "Amanda: Yes! I was thinking of trying that new Italian place.\n"
            "Hannah: Sounds great, let's meet there!"
        ),
        help=f"Supports multi-speaker dialogues up to {MAX_CHAR_LIMIT:,} characters."
    )

    # Character count counter & clear control
    char_count = len(dialogue_input)
    col_count, col_clear = st.columns([3, 1])
    col_count.caption(f"Character count: **{char_count:,}** / {MAX_CHAR_LIMIT:,}")
    if col_clear.button("Clear Input", use_container_width=True):
        st.session_state["dialogue_input"] = ""
        st.rerun()

    # Summarize Button
    summarize_pressed = st.button("Summarize Conversation", type="primary", use_container_width=True)

    if summarize_pressed:
        # Input Validation
        cleaned = clean_data(dialogue_input)

        if not cleaned:
            st.warning("⚠️ **Input Required:** Please enter or paste a conversation to summarize.")
            return

        if len(dialogue_input) > MAX_CHAR_LIMIT:
            st.error(
                f"⚠️ **Input Too Long:** The input contains {char_count:,} characters. "
                f"Please reduce it to under {MAX_CHAR_LIMIT:,} characters."
            )
            return

        if not model_loaded:
            st.error(
                "⚠️ **Model Unavailable:** The model could not be initialized. "
                "Please verify that local weights exist or provide a valid Hugging Face read token."
            )
            return

        # Inference
        try:
            with st.spinner("Synthesizing summary using beam search (4 beams)..."):
                summary = summarize_conversation(dialogue_input, model, tokenizer, device)

            # Display Result
            st.markdown("---")
            st.markdown("### 📋 Generated Summary")
            st.success(summary)

            # Copy-friendly block
            st.caption("Click the copy icon on the top right of the code box below to copy:")
            st.code(summary, language=None)

            # Metrics / reduction stats
            input_words = len(dialogue_input.split())
            summary_words = len(summary.split())
            compression = round((1 - (summary_words / max(input_words, 1))) * 100, 1)

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric("Input Words", input_words)
            stat_col2.metric("Summary Words", summary_words)
            stat_col3.metric("Reduction", f"{max(compression, 0.0)}%")

        except Exception as ex:
            logger.exception("Summarization error")
            st.error(f"❌ **Summarization Failed:** {str(ex)}")


if __name__ == "__main__":
    main()
