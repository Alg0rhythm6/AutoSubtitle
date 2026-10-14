'''
Author: Alg0rhythm6 Alg0rhythm6@outlook.com
Date: 2026-06-22 23:49:05
LastEditors: Alg0rhythm6 Alg0rhythm6@outlook.com
LastEditTime: 2026-10-11 17:09:40
FilePath: /AutoSubtitle/main.py
Description: Entry point for AutoSubtitle. Loads configuration from .env, transcribes
             video files using Whisper, translates subtitles via OpenAI API, and burns
             bilingual subtitles into the output video using FFmpeg.

Copyright (c) 2026 by Alg0rhythm6, All Rights Reserved. 
'''
import os
import gc
from dotenv import load_dotenv
import asyncio
import shutil
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe, translate, and optionally burn subtitles.")
    parser.add_argument("--mode", choices=("all", "transcribe", "translate"), default=None,
                        help="all: transcribe and translate; transcribe: generate SRT only; translate: translate existing SRT files")
    parser.add_argument("files", nargs="*", help="Video files (transcribe/all) or SRT files (translate)")
    parser.add_argument("-s", "--source", dest="source_language", help="Source language")
    parser.add_argument("-t", "--target", dest="target_language", help="Target language")
    parser.add_argument("-m", "--model", dest="openai_model", help="OpenAI translation model")
    parser.add_argument("--whisper-model", help="Whisper model: tiny/base/small/medium/large/turbo")
    parser.add_argument("--hw", "--video-hw-accel", dest="video_hw_accel", help="FFmpeg encoder: cpu/nvidia/amd/intel/mac")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), help="Whisper device (default: auto)")
    parser.add_argument("-o", "--output", dest="output_mode", choices=("both", "video", "subtitle"),
                        help="Output mode: both, video, or subtitle (default: both)")
    return parser.parse_args(argv)

def ensure_ffmpeg_available() -> None:
    """Fail fast if ffmpeg is not installed or not in PATH."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg command not found. Please install ffmpeg and ensure it is available in PATH."
        )
    
async def main(argv=None, direct_mode=False):
    args = parse_args(argv)
    current_dir = os.getcwd()
    env_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    model = args.openai_model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
    openai_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    whisper_model = args.whisper_model or os.getenv("WHISPER_MODEL", "turbo")
    source_language = args.source_language or os.getenv("SOURCE_LANGUAGE", "English")
    target_language = args.target_language or os.getenv("TARGET_LANGUAGE", "Chinese")
    output_mode = args.output_mode or os.getenv("OUTPUT_MODE", "both")
    if output_mode not in ("both", "video", "subtitle"):
        raise ValueError("OUTPUT_MODE must be one of: both, video, subtitle")
    same_language = source_language.strip().casefold() == target_language.strip().casefold()
    mode = args.mode or ("transcribe" if output_mode == "subtitle" else "all")
    if mode == "translate" and same_language:
        raise ValueError("SOURCE_LANGUAGE and TARGET_LANGUAGE are the same; translation is unnecessary.")
    if mode in ("all", "translate") and not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for translation.")
    translation_chunk_size = int(os.getenv("TRANSLATION_CHUNK_SIZE", 20))
    translation_overlap = int(os.getenv("TRANSLATION_OVERLAP", 5))
    subtitle_primary_size = int(os.getenv("SUBTITLE_PRIMARY_SIZE", 18))
    subtitle_secondary_size = int(os.getenv("SUBTITLE_SECONDARY_SIZE", 14))
    subtitle_primary_lang = os.getenv("SUBTITLE_PRIMARY_LANG", "source")
    subtitle_show_secondary = os.getenv("SUBTITLE_SHOW_SECONDARY", "true").lower() == "true"
    video_hw_accel = args.video_hw_accel or os.getenv("VIDEO_HW_ACCEL", "cpu")
    needs_video = mode in ("all", "transcribe")
    if needs_video:
        import torch
        import whisper
        from whisper.utils import get_writer
    requested_device = args.device or os.getenv("WHISPER_DEVICE", "auto")
    if needs_video and requested_device == "auto":
        if torch.cuda.is_available():
            whisper_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            whisper_device = "mps"
        else:
            whisper_device = "cpu"
    else:
        whisper_device = requested_device

    needs_ffmpeg = mode == "all" and output_mode in ("both", "video")
    if needs_ffmpeg:
        ensure_ffmpeg_available()
    if mode in ("all", "translate"):
        from Src.translator import print_env_variables, translator
        print_env_variables(api_key, model, openai_url)
    if needs_video:
        print(f"Using Whisper Model: {whisper_model}")
        print(f"Using Whisper Device: {whisper_device}")
        print(f"Source Language: {source_language}")
    print(f"Mode: {mode}")



    #get current working directory


    if mode == "translate":
        srt_path_list = ([os.path.abspath(os.path.expanduser(path)) for path in args.files]
                         if args.files else
                         [os.path.join(current_dir, "Srt", f) for f in os.listdir(os.path.join(current_dir, "Srt"))
                          if f.lower().endswith(".srt") and "_" not in os.path.splitext(f)[0]])
        if not srt_path_list:
            raise FileNotFoundError("No SRT files found. Pass SRT paths or place original SRT files in Srt/.")
        await translator(api_key, model, openai_url, [None] * len(srt_path_list), srt_path_list,
                         target_language, chunk_size=translation_chunk_size, overlap=translation_overlap)
        print("Finished translation process.")
        return

    if direct_mode:
        video_files = ([os.path.abspath(os.path.expanduser(path)) for path in args.files]
                       if args.files else
                       [os.path.join(current_dir, f) for f in os.listdir(current_dir)
                        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))])
    else:
        video_dir = os.path.join(current_dir, "Video")
        os.makedirs(video_dir, exist_ok=True)
        video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                       if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))]
    if not video_files:
        raise FileNotFoundError("No video files found.")
    srt_dir = os.path.join(current_dir, "Srt")
    if not direct_mode:
        os.makedirs(srt_dir, exist_ok=True)
    print("srt_dir:", srt_dir)

    print("Video files found:", video_files)  

    whisper_model_instance = whisper.load_model(whisper_model, device=whisper_device)

    srt_path_list = []
    for video_file in video_files:

        result = whisper_model_instance.transcribe(
            video_file,
            language=source_language,
            # MPS can be unstable with fp16 on some macOS/PyTorch versions.
            fp16=(whisper_device == "cuda"),
        )

        video_folder = (os.path.join(os.path.dirname(video_file), os.path.splitext(os.path.basename(video_file))[0])
                        if direct_mode else srt_dir)
        os.makedirs(video_folder, exist_ok=True)
        srt_writer = get_writer("srt", video_folder)
        srt_path = os.path.join(video_folder, os.path.splitext(os.path.basename(video_file))[0] + f"_{source_language}" + ".srt")
        srt_writer(result, srt_path)
        srt_path_list.append(srt_path)

    print("SRT files generated:", srt_path_list)

    # Release Whisper model to free GPU/RAM before translation
    del whisper_model_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Whisper model released from memory.")

    if mode == "transcribe":
        print("Transcription complete. SRT files generated; skipping translation and video output.")
        return

    output_dir = os.path.join(current_dir, "Output")
    from Src.subtitle import add_bilingual_subtitles, burn_subtitles
    if not direct_mode:
        os.makedirs(output_dir, exist_ok=True)
    print("Starting translation of SRT files...")
    if not same_language:
        await translator(api_key, model, openai_url, video_files, srt_path_list, target_language, chunk_size=translation_chunk_size, overlap=translation_overlap)
        print("Starting bilingual subtitle burn-in...")
    else:
        print("Source and target languages are the same. Skipping translation.")

    if output_mode in ("both", "video"):
        print("Burning subtitles into videos...")
        for video_file, srt_path in zip(video_files, srt_path_list):
            video_base = os.path.splitext(os.path.basename(video_file))[0]
            target_dir = os.path.dirname(srt_path) if direct_mode else output_dir
            os.makedirs(target_dir, exist_ok=True)
            suffix = "subtitled" if same_language else "bilingual"
            output_video_path = os.path.join(target_dir, f"{video_base}_{suffix}.mp4")
            if not same_language:
                # The bilingual helper creates the final video below.
                continue
            burn_subtitles(video_file, srt_path, output_video_path, hw_accel=video_hw_accel)

    if not same_language and output_mode in ("both", "video"):
        for video_file, srt_path in zip(video_files, srt_path_list):
            target_dir = os.path.dirname(srt_path) if direct_mode else output_dir
            add_bilingual_subtitles([video_file], [srt_path], target_language, target_dir,
                                    primary_size=subtitle_primary_size, secondary_size=subtitle_secondary_size,
                                    primary_lang=subtitle_primary_lang, show_secondary=subtitle_show_secondary,
                                    hw_accel=video_hw_accel)

    print("Finished translation process.")


if __name__ == "__main__":
    asyncio.run(main())
