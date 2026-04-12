import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. 路径配置
PROJECT_ROOT = '/root/autodl-tmp/project_douyin_mm'
# ⚠️ 改为了 xlsx 后缀
ORIGINAL_DATA_FILE = os.path.join(PROJECT_ROOT, 'scripts/01_data_prep/codebook.xlsx') 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'splits/split_v3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"🚀 开始执行 split_v3 (5分类) 数据集重构...")
print(f"📂 读取数据源: {ORIGINAL_DATA_FILE}")

# 2. 读取原始数据 (改用 read_excel)
if not os.path.exists(ORIGINAL_DATA_FILE):
    print(f"❌ 找不到原始数据文件: {ORIGINAL_DATA_FILE}，请确认路径或文件名是否正确！")
    exit()
    
# 读取 xlsx 文件
df = pd.read_excel(ORIGINAL_DATA_FILE)
initial_len = len(df)
print(f"📦 原始数据总量: {initial_len}")

# 3. 核心清洗操作
# (1) 删除损坏文件 (如果你表里的 id 是数字类型，转换为字符串去除空格)
bad_ids = ['10579', '1861', '3670']
df['video_id'] = df['video_id'].astype(str).str.strip()
df = df[~df['video_id'].isin(bad_ids)]

# (2) 删除 Policy (gold_label = 5)
df = df[df['final_code'] != 5]

clean_len = len(df)
print(f"🧹 清洗完毕。删除了 {initial_len - clean_len} 条无效或 Policy 数据。")
print(f"✅ 当前有效 5 分类数据总量: {clean_len}")

# 4. 划分固定的终极测试集 (5000 条)
TEST_SET_SIZE = 5000
if clean_len <= TEST_SET_SIZE:
    print(f"❌ 错误：清洗后的数据量（{clean_len}）不足以支撑 5000 条测试集！")
    exit()

# 使用 stratify 保证测试集各类别比例与大盘一致
df_train_pool, df_test = train_test_split(
    df, 
    test_size=TEST_SET_SIZE, 
    random_state=42,       # 固定随机种子，保证实验绝对可复现
    stratify=df['final_code']
)

# 按照习惯，命名为 test_main.csv
test_path = os.path.join(OUTPUT_DIR, 'test_main.csv')
df_test.to_csv(test_path, index=False)
print(f"\n🎯 终极测试集已锁定并保存至: test_main.csv (容量: {len(df_test)})")

# 5. 生成嵌套的增量训练集 (N=200 到 N=3000)
# 打乱训练母池，为嵌套切片做准备
df_train_pool = df_train_pool.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"🌊 剩余可用训练母池容量: {len(df_train_pool)}")

train_sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]

print("\n📈 开始生成严格嵌套的增量训练集...")
for size in train_sizes:
    if size > len(df_train_pool):
        print(f"⚠️ 警告: 训练母池不足，无法生成 train_{size}.csv！")
        break
    
    # 嵌套切片：每次都从头取，保证大集合包含小集合
    df_subset = df_train_pool.iloc[:size]
    
    subset_path = os.path.join(OUTPUT_DIR, f'train_{size}.csv')
    df_subset.to_csv(subset_path, index=False)
    print(f"   💾 生成 train_{size}.csv")

print(f"\n🎉 split_v3 文件夹及所有数据生成完毕！请前往 {OUTPUT_DIR} 查看。")