import os
import re
import json
import gc
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import glob
import subprocess
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ================= 1. 配置路径 (严格对齐 N=2200) =================
RELEASE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/root/autodl-tmp/project_douyin_mm")
BASE_MODEL_PATH = os.environ.get(
    "INTERNVL_MODEL_DIR",
    os.path.join(PROJECT_ROOT, "models/internvl/InternVL2-8B")
)
ARTIFACT_ROOT = Path(
    os.environ.get(
        "INTERNVL_ARTIFACT_ROOT",
        str(RELEASE_ROOT / "artifacts" / "internvl")
    )
)
LORA_PATH = os.environ.get(
    "INTERNVL_LORA_PATH",
    str(ARTIFACT_ROOT / "runs" / "internvl_lora_5class_2200")
)
VIDEO_BASE_DIR = os.environ.get(
    "VIDEO_BASE_DIR",
    os.path.join(PROJECT_ROOT, "videos/douyin/upload_pack")
)
CACHE_DIR = os.environ.get(
    "INTERNVL_CACHE_DIR",
    str(ARTIFACT_ROOT / "cache" / "frames_cache_internvl")
)
OUTPUT_EVAL_DIR = os.environ.get(
    "INTERNVL_OUTPUT_DIR",
    str(ARTIFACT_ROOT / "outputs" / "scaling_eval_repro")
)
os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TEST_FILES = {
    "MainTest": os.environ.get(
        "INTERNVL_TEST_CSV",
        str(RELEASE_ROOT / "splits" / "split_v3" / "test_main.csv")
    )
}

INSTR = (
    "你是一位严格的计算社会科学编码员。请将该政务短视频进行唯一分类。\n"
    "类别定义：0 OTHER_OR_UNCLEAR (其他)；1 PERFORMATIVE (绩效)；2 MORAL (道德)；3 PROCEDURAL (程序)；4 TECHNICAL (技术)。\n"
    "输出要求：仅返回JSON对象，包含键'label' (整数型)。不要包含任何多余解释。"
)

# ================= 2. 工具函数 =================
def build_video_path_index(base_dir):
    print("⏳ 正在扫描视频物理路径...")
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                vid_id = os.path.splitext(f)[0]
                path_map[vid_id] = os.path.join(root, f)
    return path_map

def build_transform(input_size=448):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if getattr(img, "mode", "RGB") != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def extract_frames_ffmpeg_cached_for_eval(video_path, cache_dir, fps=0.5, max_frames=8, max_pixels=360*480):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    vid_cache = os.path.join(cache_dir, video_id)
    os.makedirs(vid_cache, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(vid_cache, "frame_*.jpg")))
    if len(frames) < 2:
        scale_expr = f"scale='if(gt(iw*ih,{max_pixels}),trunc(iw*sqrt({max_pixels}/(iw*ih))/2)*2,iw)':-2"
        out_pattern = os.path.join(vid_cache, "frame_%06d.jpg")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={fps},{scale_expr}",
            "-q:v", "2",
            out_pattern
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            return None
        frames = sorted(glob.glob(os.path.join(vid_cache, "frame_*.jpg")))

    if not frames:
        return None

    if len(frames) > max_frames:
        idx = [int(round(i)) for i in np.linspace(0, len(frames) - 1, max_frames)]
        frames = [frames[i] for i in idx]

    imgs = []
    for p in frames:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        except Exception:
            continue

    if len(imgs) % 2 != 0 and len(imgs) > 0:
        imgs.append(imgs[-1].copy())

    return imgs

# ================= 3. 推理主逻辑 =================
def evaluate():
    # 增加显式路径检查，避免 from_pretrained 才报错
    required_paths = {
        "BASE_MODEL_PATH": BASE_MODEL_PATH,
        "LORA_PATH": LORA_PATH,
        "VIDEO_BASE_DIR": VIDEO_BASE_DIR,
        "CACHE_DIR": CACHE_DIR,
        "TEST_FILE": TEST_FILES["MainTest"],
    }
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} 不存在: {path}")

    GLOBAL_VIDEO_PATHS = build_video_path_index(VIDEO_BASE_DIR)

    print(f"🚀 加载基座: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    print(f"💉 挂载 LoRA (N=2200): {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH).eval()

    transform = build_transform(input_size=448)

    for test_name, csv_path in TEST_FILES.items():
        if not os.path.exists(csv_path):
            print(f"⚠️ 路径不存在: {csv_path}")
            continue

        print(f"\n📊 启动评估: {test_name}")
        df = pd.read_csv(csv_path)
        y_true, y_pred = [], []

        csv_save_path = os.path.join(
            OUTPUT_EVAL_DIR, "eval_internvl_lora_5class_2200_predictions.csv"
        )
        if os.path.exists(csv_save_path):
            os.remove(csv_save_path)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            vid = str(row.get("video_id", "")).strip()
            if (not vid or vid == "nan") and "rel_path" in df.columns:
                vid = os.path.splitext(os.path.basename(str(row["rel_path"])))[0]

            label = int(row["gold_label"] if "gold_label" in df.columns else row["final_code"])
            video_path = GLOBAL_VIDEO_PATHS.get(vid)

            pred = 0
            if video_path:
                try:
                    frames = extract_frames_ffmpeg_cached_for_eval(video_path, CACHE_DIR)
                    if frames:
                        pixel_values = torch.stack([transform(f) for f in frames]).to(torch.bfloat16).cuda()
                        with torch.no_grad():
                            response = model.chat(
                                tokenizer=tokenizer,
                                pixel_values=pixel_values,
                                question="<image>\n" + INSTR,
                                generation_config=dict(max_new_tokens=64, do_sample=False),
                                num_patches_list=None
                            )

                        match = re.search(r"\{.*\}", response, re.DOTALL)
                        if match:
                            pred = int(json.loads(match.group()).get("label", 0))
                        else:
                            digits = re.findall(r"\d", response)
                            pred = int(digits[0]) if digits else 0

                        del pixel_values
                except Exception:
                    pred = 0

            y_true.append(label)
            y_pred.append(pred)

            item_res = {
                "video_id": vid,
                "gold_label": label,
                "pred_label": pred
            }
            pd.DataFrame([item_res]).to_csv(
                csv_save_path,
                mode="a",
                index=False,
                header=not os.path.exists(csv_save_path),
                encoding="utf-8-sig"
            )

            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
                print(f"   [Progress] {idx+1}/{len(df)} | Current Acc: {accuracy_score(y_true, y_pred):.4f}")

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        report_str = f"InternVL2-8B Scaling Eval (N=2200)\nAcc: {acc:.4f}\n{report}"

        txt_save_path = os.path.join(
            OUTPUT_EVAL_DIR, "eval_internvl_lora_5class_2200_report.txt"
        )
        with open(txt_save_path, "w", encoding="utf-8") as f:
            f.write(report_str)

        print(f"✅ 评估完成！\n数据存至: {csv_save_path}\n报告存至: {txt_save_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    evaluate()
