'''
Author: Alg0rhythm6 Alg0rhythm6@outlook.com
Date: 2026-06-22 23:49:05
LastEditors: Alg0rhythm6 Alg0rhythm6@outlook.com
LastEditTime: 2026-06-23 13:33:35
FilePath: \subtitie\main.py
Description: Entry point for AutoSubtitle. Loads configuration from .env, transcribes
             video files using Whisper, translates subtitles via OpenAI API, and burns
             bilingual subtitles into the output video using FFmpeg.

Copyright (c) 2026 by Alg0rhythm6, All Rights Reserved. 
'''
import whisper
from whisper.utils import get_writer
import os
import gc
import torch
import openai
from Src.translator import print_env_variables, translator
from Src.subtitle import add_bilingual_subtitles
from dotenv import load_dotenv
import asyncio

async def main():
    current_dir = os.getcwd()
    env_path = os.path.join(current_dir, ".env")
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    openai_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")


    whisper_model = os.getenv("WHISPER_MODEL", "turbo")
    source_language = os.getenv("SOURCE_LANGUAGE", "English")
    target_language = os.getenv("TARGET_LANGUAGE", "Chinese")

    translation_chunk_size = int(os.getenv("TRANSLATION_CHUNK_SIZE", 20))
    translation_overlap = int(os.getenv("TRANSLATION_OVERLAP", 5))
    subtitle_primary_size = int(os.getenv("SUBTITLE_PRIMARY_SIZE", 18))
    subtitle_secondary_size = int(os.getenv("SUBTITLE_SECONDARY_SIZE", 14))
    subtitle_primary_lang = os.getenv("SUBTITLE_PRIMARY_LANG", "source")
    subtitle_show_secondary = os.getenv("SUBTITLE_SHOW_SECONDARY", "true").lower() == "true"
    video_hw_accel = os.getenv("VIDEO_HW_ACCEL", "cpu")

    print_env_variables(api_key, model, openai_url)

    print(f"Using Whisper Model: {whisper_model}")
    print(f"Source Language: {source_language}")
    print(f"Target Language: {target_language}")
    print("Environment variables loaded successfully.")
    print("--------------------------------")
    print("Starting translation process...")



    #get current working directory


    video_dir = os.path.join(current_dir, "Video")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    #get all mp4 files in the current working directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")] 
    video_files = [os.path.join(video_dir, f) for f in video_files]

    srt_dir = os.path.join(current_dir, "Srt")
    if not os.path.exists(srt_dir):
        os.makedirs(srt_dir)
    print("srt_dir:", srt_dir)

    print("Video files found:", video_files)  

    whisper_model_instance = whisper.load_model("turbo")

    srt_path_list = []
    for video_file in video_files:

        result = whisper_model_instance.transcribe(
            video_file,
            language="English"
        )

        srt_writer = get_writer("srt", srt_dir)
        srt_path = os.path.join(srt_dir, os.path.splitext(os.path.basename(video_file))[0] + f"_{source_language}" + ".srt")
        srt_writer(result, srt_path)
        srt_path_list.append(srt_path)

    print("SRT files generated:", srt_path_list)

    # Release Whisper model to free GPU/RAM before translation
    del whisper_model_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Whisper model released from memory.")

    print("Starting translation of SRT files...")
    await translator(api_key, model, openai_url, video_files, srt_path_list, target_language, chunk_size=translation_chunk_size, overlap=translation_overlap)

    print("Starting bilingual subtitle burn-in...")
    output_dir = os.path.join(current_dir, "Output")
    add_bilingual_subtitles(video_files, srt_path_list, target_language, output_dir,
                            primary_size=subtitle_primary_size, secondary_size=subtitle_secondary_size,
                            primary_lang=subtitle_primary_lang, show_secondary=subtitle_show_secondary,
                            hw_accel=video_hw_accel)

    print("Finished translation process.")


if __name__ == "__main__":
    asyncio.run(main())