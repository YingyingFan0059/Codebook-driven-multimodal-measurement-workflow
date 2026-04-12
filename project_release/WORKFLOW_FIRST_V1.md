# Workflow-First V1

> Legacy design document.
>
> This file records an earlier half-product planning stage. It is preserved for history, but it is no longer the active CBMA V1 roadmap.

## 1. 鐩爣瀹氫綅

杩欎笉鏄竴涓€滃甫鏁版嵁鐨勫紑婧愬垎绫诲櫒鈥濓紝鑰屾槸涓€涓?*鏈湴杩愯銆乧odebook-driven 鐨勫妯℃€佸垎鏋愬伐浣滄祦**銆?
绗竴鐗堢洰鏍囷細

- 鎶婄幇鏈夎鏂囨柟娉曟娊璞℃垚绋冲畾鐨?CLI/SDK 鍐呮牳
- 鐢ㄦ瀬钖勭殑鏈湴 UI 鍖呬竴灞傛祦绋嬪叆鍙?- 淇濇寔鐢ㄦ埛鏁版嵁銆佽棰戝拰妯″瀷閮界暀鍦ㄦ湰鍦?- 鍏堟敮鎸?1 鍒?2 涓ā鍨嬪悗绔紝涓嶈拷姹傗€滃ぇ鑰屽叏鈥?
浜у搧杈圭晫锛?
- 寮€婧愮殑鏄祦绋嬨€侀厤缃牸寮忋€佽瘎浼伴€昏緫鍜岄€傞厤鍣?- 涓嶅紑婧愮敤鎴锋暟鎹?- 涓嶆墭绠℃ā鍨嬶紝涔熶笉榛樿鎻愪緵鎶撳彇鍣?- 妯″瀷浠庡畼鏂规簮鎴栫敤鎴锋寚瀹氳矾寰勫姞杞?
---

## 2. 鍙傝€冮」鐩甫鏉ョ殑缁撴瀯鍚彂

鍙傝€?`Auto-claude-code-research-in-sleep` 鐨勫叧閿€濊矾涓嶆槸鈥滅収鎼畠鐨勭爺绌跺満鏅€濓紝鑰屾槸鍊熷畠鐨勭粍缁囨柟寮忥細

- 鍚屾椂鏀寔 `skills/` 宸ヤ綔娴佸拰鐙珛 CLI
- 浠撳簱灞傞潰鎶?`docs/`銆乣skills/`銆乣templates/`銆乣tools/`銆乣tests/` 鍒嗗紑
- 寮鸿皟鈥滄柟娉曡浼樺厛鈥濓紝鑰屼笉鏄厛鍋氶噸骞冲彴
- 鐢ㄨ杽灏佽鎵胯浇澶嶆潅娴佺▼锛岃€屼笉鏄妸鎵€鏈夐€昏緫鍫嗚繘 UI

瀵逛綘鐨勯」鐩紝搴旇钀芥垚锛?
- `skills/`锛氱粰 Claude Code / Codex 杩欑被 agent 鐢ㄧ殑 workflow 灏佽
- `CLI/SDK`锛氱湡姝ｇǔ瀹氱殑浜у搧鍐呮牳
- `UI`锛氬彧鍋氬弬鏁板～鍐欍€佽繍琛岀洃鎺с€佺粨鏋滃睍绀?
---

## 3. 绗竴鐗堝紑鍙戣€呬粨搴撶洰褰?
寤鸿涓嶈涓€寮€濮嬮噸鏋勬帀鐜版湁 `qwen2/` 鍜?`qwen3/`銆傛洿绋崇殑鍋氭硶鏄細**鍏堟柊澧?workflow 灞傦紝鎶婄幇鏈変唬鐮侀€愭鍖呰繘鍘?*銆?
寤鸿鐩綍锛?
```text
project_release/
鈹溾攢鈹€ README.md
鈹溾攢鈹€ WORKFLOW_FIRST_V1.md
鈹溾攢鈹€ docs/
鈹?  鈹溾攢鈹€ architecture.md
鈹?  鈹溾攢鈹€ cli.md
鈹?  鈹溾攢鈹€ config_spec.md
鈹?  鈹斺攢鈹€ model_backend_notes.md
鈹溾攢鈹€ configs/
鈹?  鈹溾攢鈹€ project.example.yaml
鈹?  鈹溾攢鈹€ codebook.example.yaml
鈹?  鈹溾攢鈹€ models/
鈹?  鈹?  鈹溾攢鈹€ qwen2_vl_7b_instruct.yaml
鈹?  鈹?  鈹斺攢鈹€ internvl2_8b.yaml
鈹?  鈹斺攢鈹€ policies/
鈹?      鈹斺攢鈹€ recommended_n.yaml
鈹溾攢鈹€ prompts/
鈹?  鈹溾攢鈹€ zeroshot/
鈹?  鈹?  鈹斺攢鈹€ classification.jinja2
鈹?  鈹溾攢鈹€ rule_based/
鈹?  鈹?  鈹斺攢鈹€ classification.jinja2
鈹?  鈹斺攢鈹€ shared/
鈹?      鈹溾攢鈹€ label_schema.jinja2
鈹?      鈹斺攢鈹€ output_json.jinja2
鈹溾攢鈹€ skills/
鈹?  鈹溾攢鈹€ mm-init/
鈹?  鈹?  鈹斺攢鈹€ SKILL.md
鈹?  鈹溾攢鈹€ mm-baseline/
鈹?  鈹?  鈹斺攢鈹€ SKILL.md
鈹?  鈹溾攢鈹€ mm-auto-lora/
鈹?  鈹?  鈹斺攢鈹€ SKILL.md
鈹?  鈹溾攢鈹€ mm-eval/
鈹?  鈹?  鈹斺攢鈹€ SKILL.md
鈹?  鈹斺攢鈹€ mm-error-analysis/
鈹?      鈹斺攢鈹€ SKILL.md
鈹溾攢鈹€ tools/
鈹?  鈹溾攢鈹€ migrate_legacy_runs.py
鈹?  鈹溾攢鈹€ generate_video_manifest.py
鈹?  鈹斺攢鈹€ doctor.py
鈹溾攢鈹€ workflow/
鈹?  鈹溾攢鈹€ pyproject.toml
鈹?  鈹溾攢鈹€ src/
鈹?  鈹?  鈹斺攢鈹€ cbma/
鈹?  鈹?      鈹溾攢鈹€ __init__.py
鈹?  鈹?      鈹溾攢鈹€ cli.py
鈹?  鈹?      鈹溾攢鈹€ sdk/
鈹?  鈹?      鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ project.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ pipeline.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ reports.py
鈹?  鈹?      鈹溾攢鈹€ core/
鈹?  鈹?      鈹?  鈹溾攢鈹€ config.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ logging.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ paths.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ errors.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ events.py
鈹?  鈹?      鈹溾攢鈹€ domain/
鈹?  鈹?      鈹?  鈹溾攢鈹€ codebook.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ dataset.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ splits.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ labels.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ metrics.py
鈹?  鈹?      鈹溾攢鈹€ storage/
鈹?  鈹?      鈹?  鈹溾攢鈹€ manifest.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ run_store.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ artifact_store.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ cache_store.py
鈹?  鈹?      鈹溾攢鈹€ prompts/
鈹?  鈹?      鈹?  鈹溾攢鈹€ compiler.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ templates.py
鈹?  鈹?      鈹溾攢鈹€ models/
鈹?  鈹?      鈹?  鈹溾攢鈹€ registry.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ downloader.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ base.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ backends/
鈹?  鈹?      鈹?      鈹溾攢鈹€ qwen2_vl.py
鈹?  鈹?      鈹?      鈹斺攢鈹€ internvl2.py
鈹?  鈹?      鈹溾攢鈹€ baselines/
鈹?  鈹?      鈹?  鈹溾攢鈹€ zeroshot.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ rule_based.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ runner.py
鈹?  鈹?      鈹溾攢鈹€ training/
鈹?  鈹?      鈹?  鈹溾攢鈹€ dataset_builder.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ lora_runner.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ sweep.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ recommended_n.py
鈹?  鈹?      鈹溾攢鈹€ evaluation/
鈹?  鈹?      鈹?  鈹溾攢鈹€ predictor.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ scorer.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ confusion.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ error_cases.py
鈹?  鈹?      鈹溾攢鈹€ reporting/
鈹?  鈹?      鈹?  鈹溾攢鈹€ markdown.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ json_report.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ export_bundle.py
鈹?  鈹?      鈹溾攢鈹€ pipelines/
鈹?  鈹?      鈹?  鈹溾攢鈹€ validate.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ baseline.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ auto_lora.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ evaluate.py
鈹?  鈹?      鈹?  鈹溾攢鈹€ report.py
鈹?  鈹?      鈹?  鈹斺攢鈹€ full_run.py
鈹?  鈹?      鈹斺攢鈹€ ui_api/
鈹?  鈹?          鈹溾攢鈹€ app.py
鈹?  鈹?          鈹溾攢鈹€ jobs.py
鈹?  鈹?          鈹斺攢鈹€ schemas.py
鈹?  鈹斺攢鈹€ tests/
鈹?      鈹溾攢鈹€ test_codebook.py
鈹?      鈹溾攢鈹€ test_dataset.py
鈹?      鈹溾攢鈹€ test_recommended_n.py
鈹?      鈹斺攢鈹€ test_cli_smoke.py
鈹溾攢鈹€ app/
鈹?  鈹斺攢鈹€ local_web/
鈹?      鈹溾攢鈹€ README.md
鈹?      鈹溾攢鈹€ static/
鈹?      鈹斺攢鈹€ templates/
鈹溾攢鈹€ qwen2/
鈹?  鈹斺攢鈹€ ...
鈹斺攢鈹€ qwen3/
    鈹斺攢鈹€ ...
```

---

鏈枃浠朵互涓嬪唴瀹逛负鏃╂湡瑙勫垝瀛樻。锛屼繚鐣欏師鏍风敤浜庡巻鍙插弬鑰冦€?