#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 评估用预处理：为测试集生成 16 秒音视频片段 (clip16)。
产出：video_clips_16s、audio_clips_16s、merged_clips_16s 及状态 CSV。
评估前需先运行本脚本，再运行 eval_qwen3_from_clips.py。
"""

import os
import sys
import subprocess
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project_douyin_mm")
CSV_PATH = os.environ.get(
    "QWEN3_TEST_CSV",
    f"{PROJECT_ROOT}/splits/split_v3/test_main.csv"
)
VIDEO_BASE_DIR = os.environ.get(
    "VIDEO_BASE_DIR",
    f"{PROJECT_ROOT}/videos/douyin/upload_pack"
)
AUDIO_BASE_DIR = os.environ.get(
    "AUDIO_BASE_DIR",
    f"{PROJECT_ROOT}/audios"
)

OUT_ROOT = os.environ.get(
    "QWEN3_EVAL_DIR",
    f"{PROJECT_ROOT}/outputs/qwen3_omni_eval_clip16_wav"
)
VIDEO_CLIP_DIR = f"{OUT_ROOT}/video_clips_16s"
AUDIO_CLIP_DIR = f"{OUT_ROOT}/audio_clips_16s"
MERGED_CLIP_DIR = f"{OUT_ROOT}/merged_clips_16s"
LOG_PATH = f"{OUT_ROOT}/prepare_clip16.log"
SUMMARY_CSV = f"{OUT_ROOT}/prepare_clip16_status.csv"

CLIP_SECONDS = 16
PRINT_EVERY = 50

VIDEO_TIMEOUT = 180
AUDIO_TIMEOUT = 120
MERGE_TIMEOUT = 120
REPAIR_TIMEOUT = 300

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(VIDEO_CLIP_DIR, exist_ok=True)
os.makedirs(AUDIO_CLIP_DIR, exist_ok=True)
os.makedirs(MERGED_CLIP_DIR, exist_ok=True)


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        log(f"[ERROR] {name} 不存在: {path}")
        sys.exit(1)


def safe_remove(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def run_cmd_quiet(cmd, timeout_sec=None):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_sec
        )
        return result.returncode == 0, "ok"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, "exception"


def build_path_index(base_dir: str, exts):
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(exts):
                stem = os.path.splitext(f)[0]
                path_map[stem] = os.path.join(root, f)
    return path_map


def cut_video_clip(src_video: str, dst_video: str, seconds: int = 16):
    if os.path.exists(dst_video) and os.path.getsize(dst_video) > 0:
        return True, "exists"

    cmd_copy = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", src_video,
        "-t", str(seconds),
        "-an",
        "-c:v", "copy",
        dst_video
    ]
    ok, status = run_cmd_quiet(cmd_copy, timeout_sec=VIDEO_TIMEOUT)
    if ok and os.path.exists(dst_video) and os.path.getsize(dst_video) > 0:
        return True, "copy"

    safe_remove(dst_video)

    cmd_reencode = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", src_video,
        "-t", str(seconds),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst_video
    ]
    ok2, status2 = run_cmd_quiet(cmd_reencode, timeout_sec=VIDEO_TIMEOUT)
    if ok2 and os.path.exists(dst_video) and os.path.getsize(dst_video) > 0:
        return True, "reencode"

    safe_remove(dst_video)
    return False, f"video_failed_{status2}"


def cut_audio_clip(src_audio: str, dst_audio: str, seconds: int = 16):
    if os.path.exists(dst_audio) and os.path.getsize(dst_audio) > 0:
        return True, "exists"

    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", src_audio,
        "-t", str(seconds),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        dst_audio
    ]
    ok, status = run_cmd_quiet(cmd, timeout_sec=AUDIO_TIMEOUT)
    if ok and os.path.exists(dst_audio) and os.path.getsize(dst_audio) > 0:
        return True, "ok"

    safe_remove(dst_audio)
    return False, f"audio_failed_{status}"


def extract_audio_from_video(src_video: str, dst_audio: str, seconds: int = 16):
    if os.path.exists(dst_audio) and os.path.getsize(dst_audio) > 0:
        return True, "exists"

    cmd = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", src_video,
        "-t", str(seconds),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        dst_audio
    ]
    ok, status = run_cmd_quiet(cmd, timeout_sec=AUDIO_TIMEOUT)
    if ok and os.path.exists(dst_audio) and os.path.getsize(dst_audio) > 0:
        return True, "ok"

    safe_remove(dst_audio)
    return False, f"audio_extract_failed_{status}"


def merge_video_audio(video_clip: str, audio_clip: str, merged_mp4: str):
    if os.path.exists(merged_mp4) and os.path.getsize(merged_mp4) > 0:
        return True, "exists"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_clip,
        "-i", audio_clip,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        merged_mp4
    ]
    ok, status = run_cmd_quiet(cmd, timeout_sec=MERGE_TIMEOUT)
    if ok and os.path.exists(merged_mp4) and os.path.getsize(merged_mp4) > 0:
        return True, "ok"

    safe_remove(merged_mp4)
    return False, f"merge_failed_{status}"


def robust_repair_one(vid: str, src_video: str, video_clip_path: str, audio_clip_path: str, merged_clip_path: str):
    safe_remove(video_clip_path)
    safe_remove(audio_clip_path)
    safe_remove(merged_clip_path)

    cmd_video = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", src_video,
        "-t", str(CLIP_SECONDS),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        video_clip_path
    ]
    ok_v, _ = run_cmd_quiet(cmd_video, timeout_sec=REPAIR_TIMEOUT)
    if not (ok_v and os.path.exists(video_clip_path) and os.path.getsize(video_clip_path) > 0):
        safe_remove(video_clip_path)
        return False, "repair_video_failed"

    cmd_audio = [
        "ffmpeg", "-y",
        "-ss", "0",
        "-i", src_video,
        "-t", str(CLIP_SECONDS),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio_clip_path
    ]
    ok_a, _ = run_cmd_quiet(cmd_audio, timeout_sec=REPAIR_TIMEOUT)
    if not (ok_a and os.path.exists(audio_clip_path) and os.path.getsize(audio_clip_path) > 0):
        safe_remove(audio_clip_path)
        return False, "repair_audio_failed"

    cmd_merge = [
        "ffmpeg", "-y",
        "-i", video_clip_path,
        "-i", audio_clip_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "16000",
        "-ac", "1",
        "-shortest",
        "-movflags", "+faststart",
        merged_clip_path
    ]
    ok_m, _ = run_cmd_quiet(cmd_merge, timeout_sec=REPAIR_TIMEOUT)
    if not (ok_m and os.path.exists(merged_clip_path) and os.path.getsize(merged_clip_path) > 0):
        safe_remove(merged_clip_path)
        return False, "repair_merge_failed"

    return True, "repair_success"


def main():
    ensure_exists(CSV_PATH, "CSV_PATH")
    ensure_exists(VIDEO_BASE_DIR, "VIDEO_BASE_DIR")
    ensure_exists(AUDIO_BASE_DIR, "AUDIO_BASE_DIR")

    log("=" * 80)
    log("[INFO] 开始仅生成 clip16（不做推理）")
    log(f"[INFO] CSV_PATH: {CSV_PATH}")
    log(f"[INFO] VIDEO_BASE_DIR: {VIDEO_BASE_DIR}")
    log(f"[INFO] AUDIO_BASE_DIR: {AUDIO_BASE_DIR}")
    log(f"[INFO] OUT_ROOT: {OUT_ROOT}")
    log("=" * 80)

    df = pd.read_csv(CSV_PATH)
    if "video_id" not in df.columns:
        raise ValueError("CSV 缺少 video_id 列")
    df["video_id"] = df["video_id"].astype(str).str.strip()

    log("[INFO] 正在建立视频索引...")
    video_paths = build_path_index(VIDEO_BASE_DIR, (".mp4", ".avi", ".mov", ".mkv"))
    log(f"[INFO] 视频索引数: {len(video_paths)}")

    log("[INFO] 正在建立音频索引...")
    audio_paths = build_path_index(AUDIO_BASE_DIR, (".wav", ".mp3", ".m4a", ".flac"))
    log(f"[INFO] 音频索引数: {len(audio_paths)}")

    stats = Counter()
    records = []

    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        vid = str(row.video_id).strip()

        video_clip_path = os.path.join(VIDEO_CLIP_DIR, f"{vid}_clip{CLIP_SECONDS}s_video.mp4")
        audio_clip_path = os.path.join(AUDIO_CLIP_DIR, f"{vid}_clip{CLIP_SECONDS}s_audio.wav")
        merged_clip_path = os.path.join(MERGED_CLIP_DIR, f"{vid}_clip{CLIP_SECONDS}s.mp4")

        if os.path.exists(merged_clip_path) and os.path.getsize(merged_clip_path) > 0:
            stats["already_done"] += 1
            records.append({
                "video_id": vid,
                "status": "already_done",
                "video_clip": video_clip_path if os.path.exists(video_clip_path) else "",
                "audio_clip": audio_clip_path if os.path.exists(audio_clip_path) else "",
                "merged_clip": merged_clip_path,
            })
            if i % PRINT_EVERY == 0:
                log(
                    f"[PROGRESS] {i}/{total} | "
                    f"already_done={stats['already_done']} | "
                    f"success={stats['success']} | "
                    f"repaired={stats['repaired']} | "
                    f"missing_video={stats['missing_video']} | "
                    f"audio_failed={stats['audio_failed']} | "
                    f"video_clip_failed={stats['video_clip_failed']} | "
                    f"merge_failed={stats['merge_failed']}"
                )
            continue

        src_video = video_paths.get(vid)
        if not src_video or not os.path.exists(src_video):
            stats["missing_video"] += 1
            records.append({"video_id": vid, "status": "missing_video"})
            continue

        ok_video, video_status = cut_video_clip(src_video, video_clip_path, seconds=CLIP_SECONDS)
        if not ok_video:
            stats["video_clip_failed"] += 1
            records.append({"video_id": vid, "status": video_status})
            continue

        src_audio = audio_paths.get(vid)
        if src_audio and os.path.exists(src_audio):
            ok_audio, audio_status = cut_audio_clip(src_audio, audio_clip_path, seconds=CLIP_SECONDS)
            if ok_audio:
                stats["used_preextracted_audio"] += 1
            else:
                stats["audio_failed"] += 1
                records.append({"video_id": vid, "status": audio_status})
                continue
        else:
            ok_audio, audio_status = extract_audio_from_video(src_video, audio_clip_path, seconds=CLIP_SECONDS)
            if ok_audio:
                stats["fallback_audio_from_video"] += 1
            else:
                stats["audio_failed"] += 1
                records.append({"video_id": vid, "status": audio_status})
                continue

        ok_merge, merge_status = merge_video_audio(video_clip_path, audio_clip_path, merged_clip_path)
        if not ok_merge:
            stats["merge_failed"] += 1
            records.append({"video_id": vid, "status": merge_status})
            continue

        stats["success"] += 1
        records.append({
            "video_id": vid,
            "status": "success",
            "video_clip": video_clip_path,
            "audio_clip": audio_clip_path,
            "merged_clip": merged_clip_path
        })

        if i % PRINT_EVERY == 0:
            log(
                f"[PROGRESS] {i}/{total} | "
                f"already_done={stats['already_done']} | "
                f"success={stats['success']} | "
                f"repaired={stats['repaired']} | "
                f"missing_video={stats['missing_video']} | "
                f"audio_failed={stats['audio_failed']} | "
                f"video_clip_failed={stats['video_clip_failed']} | "
                f"merge_failed={stats['merge_failed']}"
            )

    # 自动修复失败样本
    failed_records = [r for r in records if r.get("status") not in ("success", "already_done")]
    if failed_records:
        log(f"[INFO] 开始自动修复失败样本，共 {len(failed_records)} 条")
        repaired_records = []
        for r in failed_records:
            vid = str(r["video_id"]).strip()
            src_video = video_paths.get(vid)
            if not src_video or not os.path.exists(src_video):
                repaired_records.append({"video_id": vid, "status": "repair_missing_video"})
                continue

            video_clip_path = os.path.join(VIDEO_CLIP_DIR, f"{vid}_clip{CLIP_SECONDS}s_video.mp4")
            audio_clip_path = os.path.join(AUDIO_CLIP_DIR, f"{vid}_clip{CLIP_SECONDS}s_audio.wav")
            merged_clip_path = os.path.join(MERGED_CLIP_DIR, f"{vid}_clip{CLIP_SECONDS}s.mp4")

            ok_rep, rep_status = robust_repair_one(
                vid=vid,
                src_video=src_video,
                video_clip_path=video_clip_path,
                audio_clip_path=audio_clip_path,
                merged_clip_path=merged_clip_path,
            )

            if ok_rep:
                stats["repaired"] += 1
                repaired_records.append({
                    "video_id": vid,
                    "status": "repaired",
                    "video_clip": video_clip_path,
                    "audio_clip": audio_clip_path,
                    "merged_clip": merged_clip_path,
                })
                log(f"[REPAIR] {vid} repaired")
            else:
                stats["repair_failed"] += 1
                repaired_records.append({"video_id": vid, "status": rep_status})
                log(f"[REPAIR] {vid} failed: {rep_status}")

        good_vids = {r["video_id"] for r in repaired_records if r["status"] == "repaired"}
        records = [r for r in records if r["video_id"] not in good_vids]
        records.extend(repaired_records)

    pd.DataFrame(records).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    merged_count = sum(
        1 for f in os.listdir(MERGED_CLIP_DIR)
        if f.lower().endswith(".mp4")
    )

    log("=" * 80)
    log("[INFO] clip16 生成完成")
    log(f"[INFO] total_rows: {total}")
    log(f"[INFO] already_done: {stats['already_done']}")
    log(f"[INFO] success: {stats['success']}")
    log(f"[INFO] repaired: {stats['repaired']}")
    log(f"[INFO] repair_failed: {stats['repair_failed']}")
    log(f"[INFO] missing_video: {stats['missing_video']}")
    log(f"[INFO] used_preextracted_audio: {stats['used_preextracted_audio']}")
    log(f"[INFO] fallback_audio_from_video: {stats['fallback_audio_from_video']}")
    log(f"[INFO] audio_failed: {stats['audio_failed']}")
    log(f"[INFO] video_clip_failed: {stats['video_clip_failed']}")
    log(f"[INFO] merge_failed: {stats['merge_failed']}")
    log(f"[INFO] merged_clip_count: {merged_count}")
    log(f"[INFO] summary_csv: {SUMMARY_CSV}")
    log("=" * 80)


if __name__ == "__main__":
    main()
