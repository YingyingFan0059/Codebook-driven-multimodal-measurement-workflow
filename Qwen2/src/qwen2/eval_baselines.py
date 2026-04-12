import os
import gc
import re
import glob
import subprocess
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm  # 🌟 新增 tqdm 库用于实时进度显示

# ================= 强行注入环境变量 (防系统盘打爆) =================
PROJECT_ROOT = "/root/autodl-tmp/project_douyin_mm"
os.environ["HF_HOME"] = f"{PROJECT_ROOT}/cache/hf"
os.environ["TMPDIR"] = f"{PROJECT_ROOT}/tmp"

BASE_VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos/douyin/upload_pack")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits/split_v3")
MODEL_ID = os.path.join(PROJECT_ROOT, "models/qwen/Qwen2-VL-7B-Instruct")
CACHE_DIR = os.path.join(PROJECT_ROOT, "runs/scaling_experiments/frames_cache")
EVAL_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/scaling_eval")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

# ================= 提示词配置 (The "Baseline Injectors" - Refined V2) =================
PROMPT_PATH1 = """你是一位客观的政治传播学图像分类器。请仔细观察提供的政务短视频视觉帧，评估其传达的核心社会科学意图，并将其严格归入 {0, 1, 2, 3, 4} 中的唯一类别。

【类别学术定义】
0 OTHER_OR_UNCLEAR (其他)：无明显政治或组织意图的普通信息、文艺混剪或视觉特征不连贯的废片。
1 PERFORMATIVE (绩效)：展现官方机构或执行者实质性地达成目标、执行具体且具有结果导向的物理任务（如救援行动、巡逻守卫）。
2 MORAL (道德)：通过视觉符号传递情感共鸣，塑造内部团结、坚韧品质、奉献精神或官民之间的温情互动。
3 PROCEDURAL (程序)：强调组织内部的规范、纪律、权力交接、严肃仪式或标准化流程。
4 TECHNICAL (技术)：聚焦于硬件装备的物理展示、战术动作的专业演练、或纯粹的机械与专业实力呈现。

输出要求：仅返回合法的JSON对象，包含键 'label' (整数型)。绝不输出任何推理过程或其他内容。示例：{"label": 1}"""

PROMPT_PATH2 = """你是一位受过严格训练的计算社会科学机器编码员。我们正在对政务短视频的纯视觉抽帧进行意图编码（此时无音频辅助）。请基于以下【视觉-意图映射手册】，推断视频帧的唯一类别。

【视觉-意图映射手册】
请按照以下优先级评估画面特征：
- 类别 4 TECHNICAL (技术与实战)：寻找[纯粹的武力/专业度展示]。视觉特征：密集的武器装备特写、实弹演习爆炸画面、复杂的战术队形跑位、不涉及民众的体能拉练。
- 类别 1 PERFORMATIVE (绩效展示)：寻找[真实社会任务的执行]。视觉特征：军警在边防哨所站岗的静态画面、运送救灾物资的动态过程、实际执行抓捕或抢险的现场记录（重在“在做事”）。
- 类别 2 MORAL (道德动员)：寻找[情感与精神的视觉符号]。视觉特征：军人与民众的双向互动（如拥抱、送别）、艰苦环境下的特写（如汗水、冻伤的脸）、致敬烈士的画面、温情的节日元素。
- 类别 3 PROCEDURAL (程序规范)：寻找[权力与秩序的仪式]。视觉特征：排列整齐的会议与列队、授衔/授枪仪式中的标准动作、强调严肃纪律的视觉构图。
- 类别 0 OTHER_OR_UNCLEAR (其他)：画面仅为主持人播报新闻、纯风景混剪、或者视觉线索相互矛盾无法归入上述四类。

【判别边界提示】
- 如果画面是“刻苦训练”，若是为了展示战斗力归属 4；若特写人物咬牙坚持的痛苦与毅力，侧重情感动员，归属 2。
- 如果画面是“巡逻/站岗”，这属于日常核心职责的达成，优先归属 1，而非 3 或 4。

输出要求：仅返回JSON对象，例如 {"label": 4}。严禁输出Markdown格式符、多余空格或解释性文本。"""

# ================= 工具函数 =================
def safe_extract_frames(video_path):
    vid_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(CACHE_DIR, vid_id)
    os.makedirs(out_dir, exist_ok=True)
    
    if not glob.glob(f"{out_dir}/*.jpg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, 
                        "-vf", "fps=0.5,scale='min(360,iw)':'min(480,ih)'", 
                        "-q:v", "2", f"{out_dir}/frame_%04d.jpg"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    frames = sorted(glob.glob(f"{out_dir}/*.jpg"))
    if not frames: return None
    if len(frames) > 8: 
        frames = [frames[int(i)] for i in torch.linspace(0, len(frames)-1, steps=8).tolist()]
    if len(frames) % 2 != 0: 
        frames.append(frames[-1])
    return frames

def parse_robust_json(text):
    match = re.search(r'"label"\s*:\s*(\d)', text)
    if match: return int(match.group(1))
    digits = ''.join(filter(str.isdigit, text))
    return int(digits[0]) if digits else 0

def build_video_path_index(base_dir):
    print(f"⏳ 正在扫描视频路径...")
    path_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                vid_id = os.path.splitext(f)[0]
                path_map[vid_id] = os.path.join(root, f)
    return path_map

GLOBAL_VIDEO_PATHS = build_video_path_index(BASE_VIDEO_DIR)

# ================= 核心评估逻辑 =================
def run_baseline_eval(model, processor, path_name, prompt_text, test_csv_name="test_main.csv"):
    print(f"\n{'='*50}\n🚀 启动基线评估: [{path_name}]\n{'='*50}")
    
    df = pd.read_csv(os.path.join(SPLITS_DIR, test_csv_name), encoding='utf-8-sig')
    y_true, y_pred, results = [], [], []
    
    output_csv = os.path.join(EVAL_OUT_DIR, f"eval_baseline_{path_name}_predictions.csv")
    report_txt = os.path.join(EVAL_OUT_DIR, f"eval_baseline_{path_name}_report.txt")

    missing_count = 0
    save_step = 50  

    # 🌟 增加 tqdm 进度条，实时显示预估剩余时间 (ETA) 和当前处理速度 (it/s)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"推测进度 ({path_name})", unit="vid"):
        vid = str(row['video_id']).strip()
        gold = row['gold_label'] if 'gold_label' in df.columns else row['final_code']
        
        if vid in GLOBAL_VIDEO_PATHS:
            video_path = GLOBAL_VIDEO_PATHS[vid]
        else:
            missing_count += 1
            continue
            
        frames = safe_extract_frames(video_path)
        if not frames: continue

        messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt_text}]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=40)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            out_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            
        pred = parse_robust_json(out_text)
        if pred not in [0, 1, 2, 3, 4]: pred = 0 
            
        y_true.append(gold)
        y_pred.append(pred)
        
        results.append({
            "video_id": vid,
            "gold_label": gold,
            "pred_label": pred,
            "raw_output": out_text.strip()
        })
        
        # 依然保持每 50 条后台静默存盘，不再打印满屏日志干扰 tqdm
        if len(results) % save_step == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    VALID_LABELS = [0, 1, 2, 3, 4]
    macro_f1 = f1_score(y_true, y_pred, labels=VALID_LABELS, average='macro')
    report = classification_report(y_true, y_pred, labels=VALID_LABELS)
    
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Baseline: {path_name}\nMacro-F1: {macro_f1:.4f}\n\n{report}")
        
    print(f"✅ {path_name} 评估完成！丢失视频数: {missing_count}。报告已生成。")

if __name__ == "__main__":
    print("加载模型中...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model.eval()
    
    run_baseline_eval(model, processor, "path1_zeroshot", PROMPT_PATH1)
    run_baseline_eval(model, processor, "path2_rulebased", PROMPT_PATH2)