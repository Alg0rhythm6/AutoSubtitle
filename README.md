# AutoSubtitle

Automatically transcribes video files using [Whisper](https://github.com/openai/whisper) and translates the generated SRT subtitles via OpenAI Chat API.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

## Project Structure

```
├── main.py          # Entry point
├── Src/
│   └── translator.py  # Translation logic
├── Video/           # Place input .mp4 files here
├── Srt/             # Generated SRT files (gitignored)
├── .env             # API credentials (gitignored)
└── .env.example     # Credentials template
```

## Usage

Place `.mp4` files in the `Video/` folder, then run:

```bash
python main.py
```

Translated SRT files will be saved in the `Srt/` folder alongside the originals.
