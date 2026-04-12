from __future__ import annotations

import json


def project_config_template(project_name: str) -> dict:
    return {
        "project_name": project_name,
        "schema_version": 1,
        "paths": {
            "labels": "data/labels.csv",
            "codebook": "data/codebook.yaml",
            "videos_dir": "data/videos",
            "splits_dir": "splits/split_v1",
            "cache_dir": "cache",
            "models_dir": "models",
            "runs_dir": "runs",
            "exports_dir": "exports",
        },
        "defaults": {
            "model": "qwen2-vl-7b-instruct",
            "baselines": ["zeroshot", "rule"],
            "baseline": {
                "backend": "qwen2_legacy",
                "script": "auto",
                "test_csv": "test_main.csv",
            },
            "split_strategy": {
                "mode": "stratified",
                "train": 0.7,
                "val": 0.1,
                "test": 0.2,
                "seed": 42,
            },
            "training": {
                "candidate_n": [200, 400, 800, 1200, 1600, 2200],
                "selection_metric": "macro_f1",
                "selection_rule": {
                    "type": "min_within_delta_of_best",
                    "delta": 0.01,
                },
            },
        },
    }


def codebook_template() -> dict:
    return {
        "task_name": "douyin_military_propaganda_5class",
        "labels": [
            {
                "id": 0,
                "name": "Other",
                "definition": "Content that does not fit the analytical categories or remains unclear after review.",
                "inclusion": [
                    "Pure news reading",
                    "Scenery montage",
                    "Mixed or ambiguous cues",
                ],
                "exclusion": [
                    "Clear performative, moral, procedural, or technical emphasis",
                ],
                "boundary_cases": [
                    "Use only when no single substantive category can be defended.",
                ],
                "priority_rule": "Fallback category after other classes are ruled out.",
            },
            {
                "id": 1,
                "name": "Performative",
                "definition": "Shows concrete mission execution or public-facing institutional performance.",
                "inclusion": [
                    "Patrol",
                    "Rescue transport",
                    "Emergency response",
                ],
                "exclusion": [
                    "Symbolic tribute without real task execution",
                ],
                "boundary_cases": [
                    "If mission action is central, prefer Performative over Moral.",
                ],
                "priority_rule": "Prefer when visible action and task completion dominate the clip.",
            },
            {
                "id": 2,
                "name": "Moral",
                "definition": "Uses emotional or symbolic cues to mobilize values, sacrifice, or patriotic sentiment.",
                "inclusion": [
                    "Civil-military bonding",
                    "Tribute scenes",
                    "Hardship close-ups",
                ],
                "exclusion": [
                    "Purely procedural ceremony",
                    "Purely technical weapons display",
                ],
                "boundary_cases": [
                    "If symbolism outweighs mission action, prefer Moral over Performative.",
                ],
                "priority_rule": "Prefer when affective and symbolic meaning is the main message.",
            },
            {
                "id": 3,
                "name": "Procedural",
                "definition": "Centers rules, ceremony, order, authority, or institutional ritual.",
                "inclusion": [
                    "Assemblies",
                    "Awarding ceremonies",
                    "Formal drills with ceremonial emphasis",
                ],
                "exclusion": [
                    "Combat practice where technical skill is central",
                ],
                "boundary_cases": [
                    "If ritualized order dominates, prefer Procedural.",
                ],
                "priority_rule": "Prefer when hierarchy, formality, or ritual is the organizing logic.",
            },
            {
                "id": 4,
                "name": "Technical",
                "definition": "Highlights weapons, tactics, professional skill, or combat-oriented capability.",
                "inclusion": [
                    "Weapons close-ups",
                    "Live-fire exercise",
                    "Tactical maneuvers",
                ],
                "exclusion": [
                    "Ceremony with little operational or technical detail",
                ],
                "boundary_cases": [
                    "If technical proficiency is the main takeaway, prefer Technical.",
                ],
                "priority_rule": "Prefer when expertise and operational capability are foregrounded.",
            },
        ],
    }


def labels_csv_template() -> str:
    return "video_id,video_path,label,split\n"


def project_readme_template(project_name: str) -> str:
    return (
        f"# {project_name}\n\n"
        "This project was initialized by `cbma init`.\n\n"
        "Next steps:\n\n"
        "1. Put videos under `data/videos/` or point `labels.csv` to absolute paths.\n"
        "2. Edit `data/codebook.yaml`.\n"
        "3. Fill `data/labels.csv`.\n"
        "4. Run `cbma doctor --project .`.\n"
        "5. Run `cbma validate --project .`.\n"
        "6. Run `cbma split create --project .`.\n"
        "7. Run `cbma baseline run --project . --dry-run`.\n"
    )


def gitignore_template() -> str:
    return (
        "cache/\n"
        "models/\n"
        "runs/\n"
        "exports/\n"
        "__pycache__/\n"
        "*.pyc\n"
    )


def to_json_compatible_yaml(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
