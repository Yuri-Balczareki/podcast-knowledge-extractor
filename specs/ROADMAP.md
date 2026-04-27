# Podcast Knowledge Extractor — Roadmap

## Phase 1: Ingestion Pipeline

**Goal:** Download all podcast episodes with metadata, resumably.

**Data source:** Nerdcast RSS feed at `https://feeds.megaphone.fm/JNPD6227286900` (standard podcast XML/RSS).
No web scraping needed — the feed is structured XML with all metadata and direct MP3 links.

Each `<item>` in the feed provides:
- `<title>` — episode title and number
- `<pubDate>` — publication date
- `<itunes:duration>` — duration in seconds
- `<itunes:summary>` / `<content:encoded>` — description with guest names listed
- `<enclosure url="...">` — direct MP3 download URL
- `<guid>` — unique episode identifier

**Tasks:**
- [x] Parse the RSS feed using `feedparser` or `xml.etree.ElementTree`
- [x] Extract metadata for all episodes (title, date, duration, description, guests, audio URL, GUID)
- [x] Download MP3 files from the `<enclosure>` URLs with progress tracking and retry logic (via httpx)
- [ ] ~~Store episode metadata in a local SQLite database~~ — using CSV catalog instead (`data/jovem-nerd-episodes.csv`)
- [x] Skip already-downloaded episodes on re-run (use GUID as unique key)
- [ ] Download remaining ~430 episodes (459/1053 downloaded as of 2026-04-27)

**Milestone:** All Nerdcast episodes downloaded and cataloged. *(In progress — ~44% downloaded)*

---

## Phase 2: Transcription

**Goal:** Convert audio to text with word-level timestamps.

- [x] Evaluate three Whisper engines: openai-whisper, faster-whisper, whisper.cpp
- [x] Benchmark engines with WER, memory, and timing metrics (`scripts/compare_whisper.py`)
- [x] Unified `transcribe()` interface supporting all three engines via `--engine` flag
- [x] Portuguese initial prompt preserving proper nouns (Alottoni, Azaghal, NerdCast) and English loanwords
- [x] Batch transcription pipeline (`src/batch_transcribe.py`) with CSV progress tracking and resume support
- [x] `--duration-limit` flag to transcribe first N minutes only (useful for testing)
- [ ] Transcribe all downloaded episodes (10/459 transcribed as of 2026-04-27)
- [ ] ~~Persist transcripts to SQLite~~ — using JSON/TXT files in `data/transcripts/`

**Milestone:** All episodes transcribed with word-level timestamps. *(In progress — engine evaluation complete, bulk transcription pending)*

---

## Phase 3: Speaker Diarization

**Goal:** Identify who is speaking at each moment and track speakers across episodes.

### Evaluated: pyannote.audio
- [x] Implemented diarization module (`src/diarize.py`) using pyannote.audio
- [x] Temporal overlap merge algorithm to assign speakers to transcript segments
- [x] Integrated diarization into batch pipeline (`--diarize`, `--skip-transcription --diarize`)
- [x] Fixed tensor size mismatch on last audio window
- [x] Unit tests for merge algorithm
- [ ] **Result: quality not sufficient for production use** — keeping integration code but not using diarization in the main pipeline for now

### Cross-Episode Speaker Re-identification *(not started)*
- [ ] Extract speaker embeddings using pyannote.audio's embedding model
- [ ] Cluster embeddings across all episodes to identify recurring voices
- [ ] Build a speaker registry: manual labeling maps cluster IDs to real names (Jovem Nerd, Azaghal, etc.)

### Stretch Goal: LLM-Assisted Identification *(not started)*
- [ ] Use a local LLM to infer speaker names from conversational context
- [ ] Cross-reference with episode descriptions that list guest names

**Milestone:** Transcripts annotated with speaker labels; recurring hosts identified by name. *(Paused — diarization quality insufficient, revisit later)*

---

## Phase 4: Indexing & Vector Store *(next up)*

**Goal:** Make the full podcast corpus semantically searchable.

**Prerequisites:** All episodes transcribed (Phase 2 complete).

### 4.1 Transcript Chunking
- [ ] Split diarized JSON segments into ~500-token chunks with ~10% overlap
- [ ] Group adjacent segments by same speaker to preserve conversation flow
- [ ] Preserve metadata per chunk: episode title, episode number, speaker, start/end timestamps, air date

### 4.2 Embedding Generation
- [ ] Load multilingual sentence-transformers model (`paraphrase-multilingual-MiniLM-L12-v2`)
- [ ] Auto-detect device (MPS > CUDA > CPU), reuse existing pattern from transcribe/diarize
- [ ] Embed all chunks per episode in batch

### 4.3 ChromaDB Storage
- [ ] Persistent collection in `data/chroma/`
- [ ] Store vectors + full text + rich metadata (episode, speaker, timestamps, date)
- [ ] Add `indexing_status` column to episode CSV for progress tracking
- [ ] Skip already-indexed episodes on re-run

### 4.4 Hybrid Search
- [ ] Semantic similarity via ChromaDB vector search
- [ ] Keyword matching via BM25 (`rank-bm25`)
- [ ] Combined ranking (e.g., 70% semantic + 30% BM25)
- [ ] Return top-K results with source attribution (episode, timestamp, speaker)

### 4.5 CLI & Batch Pipeline
- [ ] `src/indexer.py` as standalone CLI script (following existing module pattern)
- [ ] Index single episode: `python src/indexer.py --episode <transcript.json>`
- [ ] Index all pending: `python src/indexer.py --all`
- [ ] Full reindex: `python src/indexer.py --reindex`
- [ ] Quick search test: `python src/indexer.py --search "query"`

### 4.6 Tests
- [ ] Unit tests for chunking logic (edge cases: empty segments, very long segments, single-speaker episodes)
- [ ] Unit tests for ChromaDB CRUD operations
- [ ] Unit tests for hybrid search ranking

**Dependencies:** `chromadb>=0.5`, `sentence-transformers>=3.3`, `rank-bm25>=0.2` (already in requirements.txt)

**Milestone:** Full corpus indexed and searchable via both semantic and keyword queries.

---

## Phase 5: Q&A Interface

**Goal:** Chat with the podcast knowledge base using a local LLM.

- [ ] RAG pipeline: user query → retrieve relevant transcript chunks → LLM generates answer with context
- [ ] Local LLM inference via Ollama with Metal acceleration on Apple Silicon
- [ ] Bilingual support: system detects query language and responds in kind
- [ ] Streamlit web interface with chat UI
- [ ] Source citations: every answer shows episode name, timestamp, and speaker

**Milestone:** Working Q&A interface with source attribution and bilingual responses.

---

## Phase 6: Future Enhancements

- [ ] Automatic topic extraction and episode tagging
- [ ] Timeline visualization of topics across episodes
- [ ] Speaker statistics: talk time per episode, total appearances, speaking patterns
- [ ] Export knowledge base to Obsidian or Notion
- [ ] Fine-tuned speaker identification model trained on labeled Nerdcast audio
- [ ] Sentiment analysis per speaker and topic

---

## TODOs

- [ ] Create `ground_truth_1025.json` fixture so functional tests can run (currently all 8 tests skip)
- [ ] Decide on diarization strategy: improve pyannote quality, try WhisperX, or defer to Phase 3 re-evaluation
