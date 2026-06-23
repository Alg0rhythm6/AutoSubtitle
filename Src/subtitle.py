import os
import srt
import subprocess


def merge_bilingual_srt(original_srt_path, translated_srt_path, output_srt_path,
                        primary_size=18, secondary_size=14, primary_lang="source",
                        show_secondary=True):
    """Merge original and translated SRT into a bilingual SRT (original / translation).
    
    primary_lang: 'source' puts original on top, 'target' puts translation on top.
    """
    with open(original_srt_path, "r", encoding="utf-8") as f:
        original_subs = list(srt.parse(f.read()))
    with open(translated_srt_path, "r", encoding="utf-8") as f:
        translated_subs = list(srt.parse(f.read()))

    translated_by_index = {sub.index: sub for sub in translated_subs}

    bilingual_subs = []
    for sub in original_subs:
        translated = translated_by_index.get(sub.index)
        source_line = f"{{\\fs{primary_size}}}{sub.content}"
        if translated and show_secondary:
            target_line = f"{{\\fs{secondary_size}}}{translated.content}"
            if primary_lang == "target":
                content = f"{target_line}\n{source_line}"
            else:
                content = f"{source_line}\n{target_line}"
        else:
            content = source_line
        bilingual_subs.append(srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=content,
        ))

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(bilingual_subs, reindex=False))

    print(f"Bilingual SRT saved to: {output_srt_path}")
    return output_srt_path


def _ffmpeg_escape_path(path):
    """Escape a file path for use in ffmpeg subtitle filter (Windows-compatible)."""
    # Use forward slashes; escape colon after drive letter and any remaining colons
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        path = path[0] + "\\:" + path[2:]
    return path


def burn_subtitles(video_path, srt_path, output_video_path):
    """Burn subtitles into video using ffmpeg."""
    srt_escaped = _ffmpeg_escape_path(srt_path)
    vf = (
        f"subtitles='{srt_escaped}'"
        ":force_style='FontName=Arial,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,Bold=0,Outline=1,Shadow=1,"
        "MarginV=20'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_video_path,
    ]

    print(f"Burning subtitles into: {os.path.basename(output_video_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"Video with bilingual subtitles saved to: {output_video_path}")
    return output_video_path


def add_bilingual_subtitles(video_files, srt_path_list, target_language, output_dir,
                            primary_size=18, secondary_size=14, primary_lang="source",
                            show_secondary=True):
    """Merge translations into bilingual SRT files and burn them into the videos."""
    os.makedirs(output_dir, exist_ok=True)

    for video_file, original_srt_path in zip(video_files, srt_path_list):
        srt_base = os.path.splitext(os.path.basename(original_srt_path))[0]
        srt_folder = os.path.dirname(original_srt_path)

        translated_srt_path = os.path.join(srt_folder, f"{srt_base}_{target_language}.srt")
        if not os.path.exists(translated_srt_path):
            print(f"Translated SRT not found, skipping: {translated_srt_path}")
            continue

        bilingual_srt_path = os.path.join(srt_folder, f"{srt_base}_bilingual.srt")
        merge_bilingual_srt(original_srt_path, translated_srt_path, bilingual_srt_path,
                            primary_size=primary_size, secondary_size=secondary_size,
                            primary_lang=primary_lang, show_secondary=show_secondary)

        video_base = os.path.splitext(os.path.basename(video_file))[0]
        output_video_path = os.path.join(output_dir, f"{video_base}_bilingual.mp4")
        burn_subtitles(video_file, bilingual_srt_path, output_video_path)
