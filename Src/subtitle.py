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


# Maps hw_accel value to (h264_encoder, hevc_encoder, quality_flags)
_HW_ENCODER_MAP = {
    "cpu":    ("libx264",            "libx265",            lambda br: ["-b:v", br, "-bufsize", str(int(br)*2)] if br else ["-crf", "18"]),
    "nvidia": ("h264_nvenc",         "hevc_nvenc",         lambda br: ["-b:v", br, "-bufsize", str(int(br)*2)] if br else ["-rc", "vbr", "-cq", "18"]),
    "amd":    ("h264_amf",           "hevc_amf",           lambda br: ["-b:v", br, "-bufsize", str(int(br)*2)] if br else ["-quality", "quality"]),
    "intel":  ("h264_qsv",           "hevc_qsv",           lambda br: ["-b:v", br, "-bufsize", str(int(br)*2)] if br else ["-global_quality", "18"]),
    "mac":    ("h264_videotoolbox",  "hevc_videotoolbox",  lambda br: ["-b:v", br] if br else ["-b:v", "8000k"]),
}


def _probe_video(video_path):
    """Return (codec_name, bit_rate) of the first video stream via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,bit_rate",
        "-of", "default=noprint_wrappers=1",
        video_path,
    ], capture_output=True, text=True)
    codec, bitrate = None, None
    for line in result.stdout.splitlines():
        if line.startswith("codec_name="):
            codec = line.split("=", 1)[1].strip()
        elif line.startswith("bit_rate="):
            val = line.split("=", 1)[1].strip()
            if val and val != "N/A":
                bitrate = val
    return codec, bitrate


def burn_subtitles(video_path, srt_path, output_video_path, hw_accel="cpu"):
    """Burn subtitles into video using ffmpeg, preserving the original video quality.
    
    hw_accel: 'cpu' | 'nvidia' | 'amd' | 'intel'
    """
    srt_escaped = _ffmpeg_escape_path(srt_path)
    vf = (
        f"subtitles='{srt_escaped}'"
        ":force_style='FontName=Arial,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,Bold=0,Outline=1,Shadow=1,"
        "MarginV=20'"
    )

    codec, bitrate = _probe_video(video_path)

    hw = hw_accel.lower() if hw_accel.lower() in _HW_ENCODER_MAP else "cpu"
    h264_enc, hevc_enc, quality_flags = _HW_ENCODER_MAP[hw]
    encoder = hevc_enc if codec == "hevc" else h264_enc

    def _build_cmd(enc, qflags):
        return ["ffmpeg", "-y", "-i", video_path, "-vf", vf,
                "-c:v", enc] + qflags + ["-c:a", "copy", output_video_path]

    print(f"Burning subtitles into: {os.path.basename(output_video_path)} (encoder: {encoder})")
    result = subprocess.run(_build_cmd(encoder, quality_flags(bitrate)), capture_output=True, text=True)

    if result.returncode != 0 and hw != "cpu":
        # GPU encoder failed — fall back to CPU automatically
        print(f"  GPU encoder ({encoder}) failed, falling back to CPU...")
        cpu_h264, cpu_hevc, cpu_quality = _HW_ENCODER_MAP["cpu"]
        cpu_encoder = cpu_hevc if codec == "hevc" else cpu_h264
        result = subprocess.run(_build_cmd(cpu_encoder, cpu_quality(bitrate)), capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    print(f"Video with bilingual subtitles saved to: {output_video_path}")
    return output_video_path


def add_bilingual_subtitles(video_files, srt_path_list, target_language, output_dir,
                            primary_size=18, secondary_size=14, primary_lang="source",
                            show_secondary=True, hw_accel="cpu"):
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
        burn_subtitles(video_file, bilingual_srt_path, output_video_path, hw_accel=hw_accel)
