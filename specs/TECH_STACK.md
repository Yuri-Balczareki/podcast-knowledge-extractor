# Podcast Knowledge Extractor — Tech Stack

  python src/transcribe.py "data/audio/NerdCast 1025 - Devoradores de Estrelas_ Rocky e Grace Salvam o Cinema.mp3" --engine whisper.cpp --model large --language pt
> Technology choices for a fully local, Apple Silicon-optimized podcast knowledge extraction pipeline.

## Core Language

**Python 3.11+** — best ecosystem for ML, audio processing, and NLP workloads.

---

## Ingestion

| Tool | Purpose | Why |
|------|---------|-----|
| **feedparser** | RSS/XML feed parsing | Standard library for podcast RSS feeds; handles namespaces (itunes, media) cleanly |
| **httpx** | Async HTTP client | Modern, async-native downloads for MP3 files with retry support |
| **SQLite** (via `sqlite3`) | Metadata & transcript storage | Zero-config, file-based, perfect for a local project |
| **tqdm** | Progress bars | Visual tracking for downloads and batch processing |

---

## Transcription & Diarization

| Tool | Purpose | Why |
|------|---------|-----|
| **faster-whisper** | Speech-to-text | 4x faster than OpenAI Whisper via CTranslate2, same accuracy. On Apple Silicon: `compute_type="float16"` with `device="cpu"` (CTranslate2 CPU is performant on M-series; MPS support is experimental) |
| **whisper.cpp** (pywhispercpp) | Speech-to-text | GGML-based C++ inference with native Metal GPU acceleration on Apple Silicon. Fastest engine for Mac hardware |
| **WhisperX** | Alignment + diarization | Wraps faster-whisper with forced phoneme alignment and pyannote.audio diarization in a single pipeline |
| **pyannote.audio** | Speaker diarization & embeddings | State-of-the-art diarization; also produces speaker embeddings for cross-episode voice matching. Requires a free HuggingFace access token |
| **FFmpeg** | Audio preprocessing | Required by Whisper for format conversion and resampling |

### Speaker Identification Strategy

**Recommended approach (pragmatic):**

1. WhisperX diarization labels segments as `SPEAKER_00`, `SPEAKER_01`, etc.
2. pyannote.audio generates voice embeddings (fingerprints) for each detected speaker
3. Cluster embeddings across all episodes to find recurring voices
4. Manual labeling: map cluster IDs to real names (Jovem Nerd, Azaghal, etc.)
5. Once labeled, new episodes auto-match speakers to known voice profiles

**Stretch goal (LLM-assisted):**

- Use a local LLM to infer speaker names from conversational context and catchphrases
- Cross-reference with episode descriptions that list guest names
- Combine contextual clues with voice embeddings for higher accuracy

**Why not train a custom voice model:**

- Requires hours of per-speaker labeled training audio
- Overkill when embedding clustering + manual labeling achieves good results
- Can revisit if the embedding-based approach proves insufficient

---

## Vector Store & Embeddings

| Tool | Purpose | Why |
|------|---------|-----|
| **ChromaDB** | Vector database | Simple, local, Python-native, no separate server process needed |
| **sentence-transformers** | Embedding generation | Local, free, multilingual models available. Recommended model: `paraphrase-multilingual-MiniLM-L12-v2` for PT-BR + EN support |
| **rank_bm25** | Keyword search | Hybrid search complement — combines BM25 keyword matching with semantic vector search for better recall |

---

## LLM & RAG

| Tool | Purpose | Why |
|------|---------|-----|
| **Ollama** | Local LLM runtime | Easiest local LLM setup, native Apple Silicon support with Metal acceleration, REST API, wide model selection |
| **LangChain** | RAG orchestration | Mature ecosystem with ChromaDB retriever, prompt templates, and chain composition. Bilingual prompt templates for PT-BR/EN |

**Recommended Ollama models:**
- `llama3.1:8b` — strong multilingual capabilities, good balance of speed and quality on Apple Silicon
- `mistral:7b` — fast inference, solid reasoning
- `sabia-2` — Portuguese-specialized (if available in Ollama registry)

---

## Interface

| Tool | Purpose | Why |
|------|---------|-----|
| **Streamlit** | Web UI | Fastest path to an interactive Q&A interface, built-in chat components, native Python |

---

## Development Tools

| Tool | Purpose | Why |
|------|---------|-----|
| **uv** | Package management | Fast, modern Python package/project manager |
| **pytest** | Testing | Standard Python test framework |
| **ruff** | Linting & formatting | Fast, replaces flake8 + black in a single tool |

---

## Project Structure

```
podcast-knowledge-extractor/
├── src/
│   ├── __init__.py
│   ├── scraper.py       # Episode fetching & async audio download
│   ├── transcriber.py      # Audio → text via faster-whisper / WhisperX
│   ├── diarizer.py         # Speaker diarization & cross-episode identification
│   ├── indexer.py          # Chunking, embedding generation, vector store
│   ├── rag.py              # RAG pipeline & Ollama LLM integration
│   └── db.py               # SQLite schema & queries
├── app/
│   └── streamlit_app.py    # Streamlit Q&A web interface
├── data/
│   ├── audio/              # Downloaded episodes (gitignored)
│   ├── transcripts/        # JSON transcripts (gitignored)
│   └── chroma/             # ChromaDB storage (gitignored)
├── tests/
├── MISSION_ROADMAP.md
├── TECH_STACK.md
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Apple Silicon Notes

- **Ollama**: Full Metal GPU acceleration out of the box — no configuration needed
- **faster-whisper**: CTranslate2 CPU backend is well-optimized for Apple Silicon's unified memory; MPS/CoreML backends are experimental
- **whisper.cpp** (pywhispercpp): Native Metal GPU acceleration, compiled in automatically on Apple Silicon — no configuration needed. Fastest transcription engine for Mac hardware
- **PyTorch** (used by pyannote.audio, sentence-transformers): MPS backend available via `device="mps"` — stable for inference, occasional issues with training
- **ChromaDB**: CPU-only, performs well for local-scale datasets (thousands of documents)
- **Memory**: `large-v3` Whisper model needs ~3GB RAM; LLM inference via Ollama will use 4-8GB depending on model size. 16GB+ Mac recommended
