# 🚀 Runqy Tasks

Ready-to-go task examples for [Runqy](https://runqy.com) — drop-in workers for common workloads.

Each task is a self-contained Runqy worker with:
- `worker.py` — Runqy worker script (`@load` / `@task` / `run()`)
- `queue.yaml` — Queue configuration for Runqy Server
- `Dockerfile` — Production container
- `requirements.txt` — Python dependencies
- `README.md` — Documentation, API examples, benchmarks

---

## 🤖 AI / ML Inference

### LLM Text Generation
> Run large language models for text generation, chat, and completion.

- **Models:** Llama 3, Mistral, Phi-3, Dolphin, Qwen (via llama.cpp GGUF)
- **Mode:** `long_running` — model stays loaded in GPU memory
- **Hardware:** GPU required (RTX 3060+ recommended)
- **Latency:** 20-50 tokens/sec depending on model & hardware
- **Use cases:** Chatbots, content generation, code completion, summarization

### Image Generation
> Generate images from text prompts using diffusion models.

- **Models:** Stable Diffusion XL, Flux, Wan2.1 (via ComfyUI or diffusers)
- **Mode:** `long_running` — model loaded once, generates on demand
- **Hardware:** GPU required (8GB+ VRAM, 12GB+ recommended)
- **Latency:** 3-15s per image depending on resolution & steps
- **Use cases:** Art generation, product mockups, marketing assets, thumbnails
- **Extras:** LoRA support, ControlNet, img2img, inpainting

### Video Generation
> Generate short video clips from text or image prompts.

- **Models:** Wan2.1/2.2 T2V & I2V (via ComfyUI or diffusers)
- **Mode:** `long_running`
- **Hardware:** GPU required (24GB+ VRAM for 14B models)
- **Latency:** 30s-5min per clip depending on length & resolution
- **Use cases:** Social media content, product demos, animated logos

### Speech-to-Text (Whisper)
> Transcribe audio and video files to text with timestamps.

- **Models:** OpenAI Whisper (tiny → large-v3), faster-whisper (CTranslate2)
- **Mode:** `long_running` or `oneshot` depending on volume
- **Hardware:** CPU viable (slower), GPU recommended for real-time
- **Latency:** 
  - faster-whisper large-v3 GPU: ~10x real-time (1min audio → 6s)
  - Whisper base CPU: ~1x real-time
- **Use cases:** Podcast transcription, meeting notes, subtitles, voice search indexing
- **Output:** Text, SRT/VTT subtitles, word-level timestamps, language detection

### Text-to-Speech
> Convert text to natural-sounding speech audio.

- **Models:** Coqui TTS, Piper (offline), Bark, XTTS-v2
- **Mode:** `long_running`
- **Hardware:** CPU viable (Piper), GPU for real-time (Coqui/Bark)
- **Latency:** 1-10s per sentence
- **Use cases:** Audiobook generation, accessibility, IVR systems, voice assistants
- **Features:** Multi-language, voice cloning (XTTS), emotion control

### Text Embeddings
> Generate vector embeddings for semantic search and RAG.

- **Models:** sentence-transformers (all-MiniLM, BGE, E5), OpenCLIP
- **Mode:** `long_running` — batch processing, high throughput
- **Hardware:** CPU viable, GPU for high volume
- **Latency:** 5-20ms per text (GPU), 20-100ms (CPU)
- **Use cases:** Semantic search, RAG pipelines, document clustering, duplicate detection, recommendation systems
- **Features:** Batch processing (hundreds of texts per request)

### Content Moderation
> Classify text and images for NSFW, violence, hate speech, and other unsafe content.

- **Models:** DiffGuard + KoalaAI + XLM-RoBERTa (text), Marqo ViT + NudeNet + CLIP (image)
- **Mode:** `long_running`
- **Hardware:** CPU only — no GPU needed
- **Latency:** ~110ms/text, ~150ms/image
- **Use cases:** User-generated content filtering, image generation gating, comment moderation
- **Features:** Multi-language (auto-translate), 9 categories, configurable thresholds, LRU cache
- **Repo:** [content-classifier](https://github.com/kitsune-hash/content-classifier)

### Zero-Shot Classification
> Classify text or images into arbitrary categories without training.

- **Models:** CLIP (images), BART-large-mnli / DeBERTa (text)
- **Mode:** `long_running`
- **Hardware:** CPU viable, GPU faster
- **Latency:** 50-200ms per item
- **Use cases:** Dynamic tagging, content routing, intent detection, image categorization

### Batch Predictions
> Run ML models on large datasets — inference at scale.

- **Models:** Any scikit-learn, PyTorch, ONNX model
- **Mode:** `oneshot` (per batch) or `long_running` (streaming)
- **Hardware:** Depends on model
- **Use cases:** Scoring leads, churn prediction, fraud detection, recommendations
- **Features:** Fan-out (split dataset → parallel workers → merge results)

---

## 🎬 Media Processing

### Video Transcoding
> Convert, compress, and reformat video files.

- **Tool:** FFmpeg
- **Mode:** `oneshot` — one job per video
- **Hardware:** CPU (hardware accel with NVENC/VAAPI if GPU available)
- **Latency:** Seconds to hours depending on length & target format
- **Use cases:** Upload processing, adaptive streaming (HLS/DASH), format conversion, compression
- **Features:** Resolution scaling, codec conversion (H.264/H.265/VP9/AV1), thumbnail extraction, audio extraction, watermarking

### Background Removal
> Remove backgrounds from images automatically.

- **Models:** rembg (U2-Net, ISNet), SAM (Segment Anything)
- **Mode:** `long_running`
- **Hardware:** CPU viable (~2-5s), GPU faster (~0.5s)
- **Latency:** 0.5-5s per image
- **Use cases:** E-commerce product photos, profile pictures, compositing, batch processing
- **Output:** PNG with transparency, mask image

### OCR / Document Text Extraction
> Extract text from images, scanned documents, and PDFs.

- **Models:** Tesseract, EasyOCR, PaddleOCR, Surya
- **Mode:** `long_running`
- **Hardware:** CPU viable, GPU for EasyOCR/PaddleOCR
- **Latency:** 1-10s per page
- **Use cases:** Invoice processing, receipt scanning, document digitization, searchable PDFs
- **Features:** 100+ languages, layout detection, table extraction, handwriting recognition

### Image Captioning
> Generate natural language descriptions of images.

- **Models:** BLIP-2, LLaVA, Florence-2
- **Mode:** `long_running`
- **Hardware:** GPU recommended (CPU possible with small models)
- **Latency:** 0.5-3s per image
- **Use cases:** Alt text generation, image search indexing, content tagging, accessibility

### Image Resize & Optimization
> Batch resize, crop, and optimize images for web delivery.

- **Tool:** Pillow, libvips
- **Mode:** `long_running` (batch) or `oneshot`
- **Hardware:** CPU only
- **Latency:** 10-100ms per image
- **Use cases:** Thumbnail generation, responsive image sets, WebP/AVIF conversion, CDN preprocessing
- **Features:** Smart crop (face-aware), format conversion, quality optimization, EXIF handling

### PDF Generation
> Generate PDF reports, invoices, and documents from templates.

- **Tools:** WeasyPrint, ReportLab, Playwright (HTML→PDF)
- **Mode:** `oneshot` or `long_running`
- **Hardware:** CPU only
- **Latency:** 0.5-5s per document
- **Use cases:** Invoice generation, report export, certificate generation, bulk mail merge

### PDF Text Extraction
> Extract structured text, tables, and metadata from PDF files.

- **Tools:** pdfplumber, PyMuPDF, Camelot (tables), Unstructured
- **Mode:** `oneshot`
- **Hardware:** CPU only
- **Latency:** 0.5-10s per document
- **Use cases:** Document parsing, contract analysis, data extraction, RAG document ingestion

### Face Detection & Blur
> Detect faces in images/video and apply blur for privacy.

- **Models:** MediaPipe, RetinaFace, YOLO-face
- **Mode:** `long_running`
- **Hardware:** CPU viable, GPU for video
- **Latency:** 20-100ms per image, real-time for video on GPU
- **Use cases:** GDPR compliance, privacy protection, video anonymization, dashcam footage

### Screenshot Generation
> Capture screenshots of web pages at various viewports.

- **Tools:** Playwright (Chromium headless)
- **Mode:** `long_running` (browser stays loaded)
- **Hardware:** CPU, 512MB+ RAM
- **Latency:** 2-10s per page (including render)
- **Use cases:** Link previews, social cards, SEO monitoring, visual regression testing, archiving
- **Features:** Full page, viewport sizes, PDF export, wait for elements, block ads

### Audio Waveform Generation
> Generate visual waveform images from audio files.

- **Tools:** FFmpeg, audiowaveform, matplotlib
- **Mode:** `oneshot`
- **Hardware:** CPU only
- **Latency:** 1-5s per file
- **Use cases:** Music players, podcast previews, audio editing UI, social media

---

## 📊 Data Analytics & ETL

### ETL Pipeline
> Extract, transform, and load data between sources.

- **Tools:** pandas, polars, DuckDB, SQLAlchemy
- **Mode:** `oneshot` (per batch/schedule)
- **Hardware:** CPU, RAM proportional to dataset size
- **Latency:** Seconds to hours depending on volume
- **Use cases:** Database sync, API→warehouse, CSV normalization, data lake ingestion
- **Features:** Schema mapping, data validation, incremental loads, error handling with dead letter queue

### Report Generation
> Aggregate data and generate formatted reports.

- **Tools:** pandas/polars + Jinja2 + WeasyPrint/Excel
- **Mode:** `oneshot` (scheduled daily/weekly)
- **Hardware:** CPU only
- **Latency:** 5s-5min depending on data volume
- **Use cases:** Daily KPI dashboards, financial reports, client reports, compliance reports
- **Output:** PDF, Excel, HTML, CSV

### Data Quality Checks
> Validate datasets against rules, detect anomalies and drift.

- **Tools:** Great Expectations, pandas, custom validators
- **Mode:** `oneshot` (post-ingestion trigger or scheduled)
- **Hardware:** CPU only
- **Latency:** Seconds to minutes
- **Use cases:** Pipeline health, schema validation, freshness checks, null/duplicate detection, statistical anomaly detection
- **Features:** Rule-based + statistical, alerting on failure, historical tracking

### Web Scraping
> Crawl and extract structured data from websites at scale.

- **Tools:** Playwright, BeautifulSoup, Scrapy
- **Mode:** `long_running` (browser pool) or `oneshot`
- **Hardware:** CPU, 512MB+ RAM per browser
- **Latency:** 2-30s per page
- **Use cases:** Price monitoring, competitor analysis, lead generation, content aggregation, market research
- **Features:** JavaScript rendering, anti-bot handling, rate limiting, proxy rotation, structured output

### Log Aggregation & Analysis
> Parse, aggregate, and analyze application logs.

- **Tools:** pandas, regex, custom parsers
- **Mode:** `oneshot` (per batch) or `long_running` (streaming)
- **Hardware:** CPU, RAM proportional to log volume
- **Latency:** Varies
- **Use cases:** Error tracking, performance monitoring, security audit, usage analytics
- **Features:** Pattern matching, anomaly detection, alerting, metric extraction

### Time Series Forecasting
> Generate forecasts from historical time series data.

- **Models:** Prophet, statsmodels (ARIMA/ETS), NeuralProphet, TimesFM
- **Mode:** `oneshot`
- **Hardware:** CPU (Prophet), GPU (NeuralProphet/TimesFM)
- **Latency:** 1-60s per series
- **Use cases:** Demand forecasting, capacity planning, financial projections, inventory management
- **Features:** Seasonality detection, holiday effects, confidence intervals, multi-series

### Data Anonymization
> Remove or mask PII (Personally Identifiable Information) from datasets.

- **Tools:** Presidio (Microsoft), spaCy NER, regex patterns
- **Mode:** `oneshot` (per dataset)
- **Hardware:** CPU only
- **Latency:** 10-100ms per record
- **Use cases:** GDPR compliance, data sharing, test data generation, analytics on sensitive data
- **Features:** Names, emails, phones, addresses, credit cards, custom patterns, consistent pseudonymization

### Database Migration
> Transform and migrate data between databases or schemas.

- **Tools:** SQLAlchemy, Alembic, pandas
- **Mode:** `oneshot`
- **Hardware:** CPU, network I/O bound
- **Latency:** Minutes to hours
- **Use cases:** Schema upgrades, platform migration, data warehouse loading, sharding
- **Features:** Incremental migration, rollback, validation, progress tracking

### RSS / Feed Aggregation
> Fetch, parse, and normalize content from RSS/Atom feeds.

- **Tools:** feedparser, httpx
- **Mode:** `oneshot` (scheduled)
- **Hardware:** CPU, minimal
- **Latency:** 1-5s per feed
- **Use cases:** News aggregation, content curation, competitive intelligence, social listening
- **Features:** Deduplication, content extraction, category tagging, digest generation

### Batch Geocoding
> Convert addresses to coordinates (and reverse) at scale.

- **Tools:** Nominatim (OpenStreetMap, self-hosted), geopy
- **Mode:** `oneshot` (per batch)
- **Hardware:** CPU only
- **Latency:** 50-500ms per address (rate limited)
- **Use cases:** Customer mapping, delivery routing, location analytics, data enrichment

---

## 📁 Repository Structure

```
runqy-tasks/
├── README.md                    # This file — task catalog
├── ai/
│   ├── llm-inference/           # LLM text generation
│   ├── image-generation/        # Stable Diffusion / Flux
│   ├── video-generation/        # Wan2.1 T2V/I2V
│   ├── whisper-transcription/   # Speech-to-text
│   ├── text-to-speech/          # TTS
│   ├── text-embeddings/         # Vector embeddings
│   ├── content-moderation/      # NSFW/safety classification
│   ├── zero-shot-classification/
│   └── batch-predictions/
├── media/
│   ├── video-transcoding/       # FFmpeg
│   ├── background-removal/      # rembg
│   ├── ocr/                     # Document text extraction
│   ├── image-captioning/        # BLIP-2
│   ├── image-resize/            # Batch resize & optimize
│   ├── pdf-generation/          # HTML/template → PDF
│   ├── pdf-extraction/          # PDF → structured text
│   ├── face-detection-blur/     # Privacy blur
│   ├── screenshot/              # URL → PNG
│   └── audio-waveform/          # Waveform visualization
├── data/
│   ├── etl-pipeline/            # Extract/Transform/Load
│   ├── report-generation/       # Aggregate → PDF/Excel
│   ├── data-quality/            # Validation & anomaly detection
│   ├── web-scraping/            # Playwright crawling
│   ├── log-analysis/            # Log aggregation
│   ├── time-series-forecast/    # Prophet / statsmodels
│   ├── data-anonymization/      # PII removal
│   ├── database-migration/      # Schema migration
│   ├── rss-aggregation/         # Feed parsing
│   └── batch-geocoding/         # Address → coordinates
└── templates/
    ├── python-oneshot/          # Starter template (oneshot)
    └── python-long-running/     # Starter template (long_running)
```

Each task directory contains:
```
task-name/
├── worker.py            # Runqy worker
├── queue.yaml           # Queue configuration
├── Dockerfile           # Production container
├── requirements.txt     # Dependencies
├── README.md            # Docs, API, benchmarks
└── examples/            # Example payloads
```

---

## 🏷️ Task Metadata

Each task will include metadata for the future Runqy marketplace:

```yaml
# task.yaml
name: whisper-transcription
display_name: "Speech-to-Text (Whisper)"
category: ai
tags: [audio, transcription, speech, whisper, subtitles]
hardware: [cpu, gpu]
mode: long_running
avg_latency: "6s per minute of audio (GPU)"
memory_mb: 3000
description: "Transcribe audio and video files to text with timestamps using OpenAI Whisper."
```

---

## Getting Started

1. Pick a task from the catalog above
2. Copy the task directory to your project
3. Configure `queue.yaml` for your Runqy Server
4. Deploy: `runqy deploy ./task-name`

See [Runqy Documentation](https://docs.runqy.com) for setup instructions.

---

## Contributing

Want to add a task? See [templates/](./templates/) for starter templates.

## License

MIT
