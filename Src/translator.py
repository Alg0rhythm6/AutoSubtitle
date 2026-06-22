from openai import OpenAI
import os
import srt

def print_env_variables(api_key, model, openai_url):
    print(f"Using OpenAI API Key: {api_key}")
    print(f"Using OpenAI Model: {model}")
    print(f"Using OpenAI API Base URL: {openai_url}")

def translator(api_key, model, openai_url, video_files, srt_path_list, target_language):
    client = OpenAI(api_key=api_key, base_url=openai_url)

    for video_file, srt_path in zip(video_files, srt_path_list):

        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
        subs = list(srt.parse(srt_content))

        prompt = (
            f"Translate the following SRT subtitles into {target_language}.\n\n"
            "Requirements:\n"
            "1. Keep the original subtitle numbers.\n"
            "2. Keep the original timestamps exactly unchanged.\n"
            "3. Only translate the subtitle text.\n"
            "4. Do not add explanations.\n"
            "5. Do not use Markdown.\n"
            "6. Return only valid SRT format.\n\n"
            f"{srt_content}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional subtitle translation assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
        )

        translated_srt = response.choices[0].message.content

        translated_srt_path = os.path.join(os.path.dirname(srt_path), os.path.splitext(os.path.basename(srt_path))[0] + f"_{target_language}.srt")
        with open(translated_srt_path, "w", encoding="utf-8") as f:
            f.write(translated_srt)

        print(f"Translated subtitles saved to: {translated_srt_path}")


    