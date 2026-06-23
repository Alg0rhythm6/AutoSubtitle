from openai import AsyncOpenAI
import os
import srt
import asyncio

def print_env_variables(api_key, model, openai_url):
    print(f"Using OpenAI API Key: {api_key}")
    print(f"Using OpenAI Model: {model}")
    print(f"Using OpenAI API Base URL: {openai_url}")

async def _translate_batch(client, model, target_language, subs_snapshot, core_start, core_end):
    """Translate one batch and return (core_start, core_end, translated_by_index)."""
    batch_srt_content = srt.compose(subs_snapshot, reindex=False)
    prompt = (
        f"Translate the following SRT subtitles into {target_language}.\n\n"
        "Requirements:\n"
        "1. Keep the original subtitle numbers.\n"
        "2. Keep the original timestamps exactly unchanged.\n"
        "3. Only translate the subtitle text.\n"
        "4. Do not add explanations.\n"
        "5. Do not use Markdown.\n"
        "6. Return only valid SRT format.\n\n"
        f"{batch_srt_content}"
    )
    response = await client.chat.completions.create(
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
    translated_batch_subs = list(srt.parse(response.choices[0].message.content))
    translated_by_index = {sub.index: sub for sub in translated_batch_subs}
    return core_start, core_end, translated_by_index

async def _translate_file(client, model, target_language, srt_path, chunk_size, overlap):
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    subs = list(srt.parse(srt_content))
    total = len(subs)

    # Build all batch tasks up front (subs is read-only during gather)
    tasks = []
    batch_ranges = []
    for i in range(0, total, chunk_size):
        core_start = i
        core_end = min(i + chunk_size, total)
        ctx_start = max(0, core_start - overlap)
        ctx_end = min(total, core_end + overlap)
        subs_snapshot = subs[ctx_start:ctx_end]  # copy slice for safe concurrent reads
        tasks.append(_translate_batch(client, model, target_language, subs_snapshot, core_start, core_end))
        batch_ranges.append((core_start, core_end))

    results = await asyncio.gather(*tasks)

    # Apply all results after gather completes (no concurrent writes)
    for core_start, core_end, translated_by_index in results:
        for j in range(core_start, core_end):
            orig_index = subs[j].index
            if orig_index in translated_by_index:
                subs[j].content = translated_by_index[orig_index].content
        print(f"  Translated {core_end}/{total} subtitles")

    translated_srt_path = os.path.join(
        os.path.dirname(srt_path),
        os.path.splitext(os.path.basename(srt_path))[0] + f"_{target_language}.srt"
    )
    with open(translated_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    print(f"Translated subtitles saved to: {translated_srt_path}")

async def translator(api_key, model, openai_url, video_files, srt_path_list, target_language, chunk_size=20, overlap=5):
    client = AsyncOpenAI(api_key=api_key, base_url=openai_url)
    await asyncio.gather(*[
        _translate_file(client, model, target_language, srt_path, chunk_size, overlap)
        for _, srt_path in zip(video_files, srt_path_list)
    ])



    