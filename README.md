# Podcast Knowledge Extractor

End-to-end pipeline for downloading podcast episodes, transcribing audio to text, and building a searchable knowledge base powered by LLMs for information extraction and conversational Q&A.

Built for **Brazilian Portuguese** content (Nerdcast podcast), fully local with no cloud APIs, and optimized for **Apple Silicon** (Metal/MPS acceleration).

## Overview

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Ingestion | Done | RSS feed parsing, async MP3 downloads, CSV episode catalog |
| 2. Transcription | Done | Audio-to-text with 3 Whisper engines behind a unified interface |
| 3. Speaker Diarization | Done | Speaker identification via pyannote.audio with temporal overlap merge |
| 4. Indexing & Vector Store | Planned | Semantic chunking, embeddings, ChromaDB hybrid search |
| 5. Q&A Interface | Planned | RAG pipeline with local LLM (Ollama) + Streamlit UI |
| 6. Knowledge Mining | Future | Topic extraction, timelines, speaker stats, Obsidian export |

## Features

- **Multi-engine transcription** — faster-whisper (CTranslate2, 4x faster), openai-whisper (reference), and whisper.cpp (native Metal GPU via GGML)
- **Speaker diarization** — pyannote.audio pipeline with O(N*M) temporal overlap merge algorithm to assign speakers to transcript segments
- **Batch downloading** — RSS feed sync to CSV catalog with resume support, parallel downloads via httpx
- **Batch transcription** — CSV-tracked pipeline with auto-detect, resume, and optional parallel workers via ProcessPoolExecutor
- **Portuguese-aware prompting** — initial prompt preserves proper nouns (Alottoni, Azaghal, NerdCast) and English loanwords
- **Benchmarking** — compare engines on WER, wall time, real-time factor, and memory usage
- **Device auto-detection** — MPS > CUDA > CPU, across all ML modules

## Tech Stack

| Layer | Tools | Purpose |
|-------|-------|---------|
| Ingestion | feedparser, httpx, tqdm | RSS parsing, async downloads, progress bars |
| Transcription | openai-whisper, faster-whisper, pywhispercpp | 3 Whisper engines, unified `transcribe()` interface |
| Diarization | pyannote.audio | Speaker identification + temporal merge |
| Vector Store | ChromaDB, sentence-transformers, rank-bm25 | Semantic + BM25 hybrid search (Phase 4) |
| LLM & RAG | Ollama, LangChain | Local LLM inference, RAG orchestration (Phase 5) |
| UI | Streamlit | Web-based Q&A interface (Phase 5) |
| Dev Tools | pytest, ruff, jiwer | Testing, linting, WER benchmarking |

## Architecture

```
RSS Feed ─> scraper.py ─> MP3 in data/audio/
                                    │
                                    v
                            transcribe.py ─> JSON/TXT in data/transcripts/
                                    │
                                    v
                             diarize.py ─> *.diarized.json/txt with speaker labels
                                    │
                                    v
                          (Phase 4) indexer ─> ChromaDB in data/chroma/
                                    │
                                    v
                          (Phase 5) RAG + Streamlit UI
```

Each module is a standalone CLI script with `argparse` and `if __name__ == "__main__": main()`.

## Project Structure

```
podcast-knowledge-extractor/
├── src/
│   ├── scraper.py          # Phase 1: RSS feed sync + batch MP3 download
│   ├── transcribe.py          # Phase 2: Multi-engine audio transcription
│   ├── batch_transcribe.py    # Phase 2: Batch transcription with CSV tracking
│   └── diarize.py             # Phase 3: Speaker diarization + merge
├── scripts/
│   └── compare_whisper.py     # Engine benchmarking (WER, timing, memory)
├── tests/
│   └── unit/
│       ├── test_diarize.py    # Merge algorithm + I/O tests
│       └── test_batch_transcribe.py  # Batch pipeline logic tests
├── app/                       # Streamlit UI (Phase 5)
├── data/
│   ├── audio/                 # Downloaded episodes (gitignored)
│   ├── transcripts/           # JSON + TXT transcripts (gitignored)
│   ├── chroma/                # Vector store (gitignored)
│   └── comparisons/           # Benchmark results (gitignored)
├── models/                    # whisper.cpp GGML binaries
├── docs/features/             # Design documents
├── MISSION_ROADMAP.md
├── TECH_STACK.md
├── CHANGELOG.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.12+
- Apple Silicon Mac recommended (MPS/Metal acceleration)
- [HuggingFace token](https://huggingface.co/settings/tokens) for speaker diarization (accept model licenses for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0))

### Setup

```bash
git clone https://github.com/yourusername/podcast-knowledge-extractor.git
cd podcast-knowledge-extractor

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file for diarization:

```
HF_TOKEN=hf_your_huggingface_token
```

For whisper.cpp, place the GGML model binary at `models/ggml-base.bin`.

## Usage

### Download episodes

```bash
# Sync RSS feed and download all pending episodes
python src/scraper.py

# Sync feed to CSV without downloading
python src/scraper.py --sync-only

# Download up to 5 pending episodes
python src/scraper.py --limit 5
```

### Transcribe audio

```bash
# Default: faster-whisper, base model
python src/transcribe.py data/audio/episode.mp3

# Choose engine and model
python src/transcribe.py data/audio/episode.mp3 --engine openai-whisper --model large --language pt
python src/transcribe.py data/audio/episode.mp3 --engine whisper.cpp --model base
```

Output: `data/transcripts/{name}.json` (segments with timestamps) and `data/transcripts/{name}.txt` (full text).

### Batch transcription

```bash
# Transcribe all downloaded episodes (sequential, whisper.cpp large model)
python src/batch_transcribe.py

# Preview what would be transcribed
python src/batch_transcribe.py --dry-run

# Transcribe up to 10 episodes with 2 parallel workers
python src/batch_transcribe.py --limit 10 --workers 2

# Use a specific model and language
python src/batch_transcribe.py --model large --language pt
```

Tracks transcription status in the episode CSV (`transcription_status`, `transcript_path`). Re-running skips already-transcribed episodes. Existing transcripts are auto-detected on first run.

### Speaker diarization

```bash
# Auto-detects matching transcript JSON
python src/diarize.py data/audio/episode.mp3

# Explicit transcript path
python src/diarize.py data/audio/episode.mp3 --transcript data/transcripts/episode.json
```

Output: `data/transcripts/{name}.diarized.json` and `data/transcripts/{name}.diarized.txt` with speaker labels.

### Benchmark engines

```bash
# Compare all engines on the same audio
python scripts/compare_whisper.py data/audio/episode.mp3 --model base --engines openai-whisper faster-whisper whisper.cpp --save-json

# Quick test on first 60 seconds
python scripts/compare_whisper.py data/audio/episode.mp3 --max-duration 60
```

## Testing

```bash
# Run all tests
pytest tests/

# Single file with verbose output
pytest tests/unit/test_diarize.py -v

# Linting
ruff check src/ tests/
ruff format src/ tests/
```

## Roadmap

**Phase 4 — Indexing & Vector Store**: semantic chunking of diarized transcripts, multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), ChromaDB storage with BM25 hybrid search.

**Phase 5 — Q&A Interface**: RAG pipeline using LangChain + Ollama (local LLM), Streamlit web UI for conversational Q&A over podcast content.

**Phase 6 — Knowledge Mining**: topic extraction, episode timelines, speaker statistics, and Obsidian vault export.

## License

[MIT](LICENSE)
