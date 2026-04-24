# Podcast Knowledge Extractor — Mission Roadmap

> Build an end-to-end pipeline that downloads Nerdcast podcast episodes, transcribes them, identifies speakers, and provides a local AI-powered Q&A interface over the entire knowledge base.

## Constraints

- **Fully local** — no cloud APIs, everything runs on-device
- **Apple Silicon optimized** — leverage Metal/MPS acceleration where available
- **Bilingual** — Q&A interface supports both Portuguese (PT-BR) and English
- **Resumable** — every phase supports picking up where it left off

---

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
- Parse the RSS feed using `feedparser` or `xml.etree.ElementTree`
- Extract metadata for all episodes (title, date, duration, description, guests, audio URL, GUID)
- Async download MP3 files from the `<enclosure>` URLs with progress tracking and retry logic
- Store episode metadata in a local SQLite database
- Skip already-downloaded episodes on re-run (use GUID as unique key)

**Milestone:** All Nerdcast episodes downloaded and cataloged in SQLite.

---

## Phase 2: Transcription

**Goal:** Convert audio to text with word-level timestamps.

- Transcribe each episode using faster-whisper with the `large-v3` model
- Language: Portuguese (`pt`) with automatic fallback detection
- Output structured JSON per episode: list of segments with `start`, `end`, `text`
- Persist transcripts to SQLite alongside episode metadata
- Skip already-transcribed episodes on re-run

**Milestone:** All episodes transcribed with word-level timestamps.

---

## Phase 3: Speaker Diarization

**Goal:** Identify who is speaking at each moment and track speakers across episodes.

### Primary Approach: WhisperX + pyannote.audio
- WhisperX combines faster-whisper transcription with pyannote.audio diarization
- Produces word-level timestamps aligned with speaker segments
- Labels speakers as `SPEAKER_00`, `SPEAKER_01`, etc. within each episode

### Cross-Episode Speaker Re-identification
- Extract speaker embeddings using pyannote.audio's embedding model
- Cluster embeddings across all episodes to identify recurring voices
- Build a speaker registry: manual labeling maps cluster IDs to real names (Jovem Nerd, Azaghal, etc.)
- Once labeled, new episodes auto-match speakers to known voice profiles

### Stretch Goal: LLM-Assisted Identification
- After transcription, use a local LLM to infer speaker names from conversational context
- Cross-reference with episode descriptions that often list guest names
- Identify speakers by characteristic phrases, catchphrases, or speech patterns

**Milestone:** Transcripts annotated with speaker labels; recurring hosts identified by name.

---

## Phase 4: Indexing & Vector Store

**Goal:** Make the full podcast corpus semantically searchable.

- Chunk transcripts into semantic segments (~500 tokens with overlap)
- Generate embeddings using a multilingual sentence-transformers model
- Store vectors in ChromaDB with rich metadata (episode, speaker, timestamp, date)
- Implement hybrid search: semantic similarity (vector) + keyword matching (BM25)

**Milestone:** Full corpus indexed and searchable via both semantic and keyword queries.

---

## Phase 5: Q&A Interface

**Goal:** Chat with the podcast knowledge base using a local LLM.

- RAG pipeline: user query → retrieve relevant transcript chunks → LLM generates answer with context
- Local LLM inference via Ollama with Metal acceleration on Apple Silicon
- Bilingual support: system detects query language and responds in kind
- Streamlit web interface with chat UI
- Source citations: every answer shows episode name, timestamp, and speaker

**Milestone:** Working Q&A interface with source attribution and bilingual responses.

---

## Phase 6: Future Enhancements

- Automatic topic extraction and episode tagging
- Timeline visualization of topics across episodes
- Speaker statistics: talk time per episode, total appearances, speaking patterns
- Export knowledge base to Obsidian or Notion
- Fine-tuned speaker identification model trained on labeled Nerdcast audio
- Sentiment analysis per speaker and topic
