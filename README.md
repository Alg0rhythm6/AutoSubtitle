# AutoSubtitle

Automatically transcribes video files using [Whisper](https://github.com/openai/whisper), translates the subtitles via OpenAI API, and burns bilingual subtitles into the output video.

## Features

- Speech-to-text transcription with Whisper (runs locally, no API cost)
- Concurrent subtitle translation via OpenAI Chat API
- Bilingual subtitle merging (original + translation)
- Configurable font sizes, language order, and secondary subtitle toggle
- Subtitles burned into output video via FFmpeg

## Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in system `PATH`
- An OpenAI-compatible API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Alg0rhythm6/AutoSubtitle.git
   cd AutoSubtitle
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

## Configuration

All options are set in the `.env` file. See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model used for translation |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | API base URL (replace for third-party proxies) |
| `WHISPER_MODEL` | `turbo` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large` / `turbo` |
| `SOURCE_LANGUAGE` | `English` | Language spoken in the video |
| `TARGET_LANGUAGE` | `Chinese` | Translation target language |
| `TRANSLATION_CHUNK_SIZE` | `20` | Subtitles per API request |
| `TRANSLATION_OVERLAP` | `5` | Context lines added around each chunk |
| `SUBTITLE_PRIMARY_SIZE` | `18` | Primary subtitle font size (pt) |
| `SUBTITLE_SECONDARY_SIZE` | `14` | Secondary subtitle font size (pt) |
| `SUBTITLE_PRIMARY_LANG` | `source` | Language on top: `source` (original) or `target` (translation) |
| `SUBTITLE_SHOW_SECONDARY` | `true` | `true` = bilingual, `false` = primary only |

## Usage

Place `.mp4` files in the `Video/` folder, then run:

```bash
python main.py
```

### Pipeline

```
Video/*.mp4
  |
  |-- Whisper transcription --> Srt/*_English.srt
  |-- OpenAI translation    --> Srt/*_English_Chinese.srt
  |-- Bilingual SRT merge   --> Srt/*_English_bilingual.srt
  `-- FFmpeg burn-in        --> Output/*_bilingual.mp4
```

## Project Structure

```
|-- main.py               # Entry point
|-- Src/
|   |-- translator.py     # Async subtitle translation
|   `-- subtitle.py       # Bilingual SRT merge and FFmpeg burn-in
|-- Video/                # Place input .mp4 files here
|-- Srt/                  # Generated SRT files (gitignored)
|-- Output/               # Output videos with burned-in subtitles (gitignored)
|-- requirements.txt      # Python dependencies
|-- .env                  # Local configuration (gitignored)
`-- .env.example          # Configuration template
```
