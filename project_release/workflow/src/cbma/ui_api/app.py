from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cbma.ui_api.service import build_page_context, run_ui_action


UI_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"


def create_app(project_root: str | Path | None = None) -> FastAPI:
    resolved_project = Path(project_root or os.environ.get("CBMA_UI_PROJECT", ".")).expanduser().resolve()

    app = FastAPI(title="CBMA Local UI")
    app.state.project_root = resolved_project
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def render(request: Request, template_name: str, current_page: str, page_title: str, action_result: dict | None = None):
        context = build_page_context(
            app.state.project_root,
            current_page=current_page,
            page_title=page_title,
            action_result=action_result,
        )
        context["request"] = request
        return templates.TemplateResponse(request, template_name, context)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        return render(request, "dashboard.html", current_page="dashboard", page_title="Research Control Room")

    @app.post("/actions/{action_name}", response_class=HTMLResponse)
    async def run_action(request: Request, action_name: str, redirect_to: str = Form("dashboard")) -> HTMLResponse:
        action_result = run_ui_action(app.state.project_root, action_name)
        template_map = {
            "dashboard": ("dashboard.html", "dashboard", "Research Control Room"),
            "split": ("split.html", "split", "Split Builder"),
            "baseline": ("baseline.html", "baseline", "Baseline Console"),
            "train": ("train.html", "train", "Sweep Planner"),
        }
        template_name, current_page, page_title = template_map.get(
            redirect_to,
            ("dashboard.html", "dashboard", "Research Control Room"),
        )
        return render(
            request,
            template_name,
            current_page=current_page,
            page_title=page_title,
            action_result=action_result,
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(request: Request) -> HTMLResponse:
        return render(request, "settings.html", current_page="settings", page_title="Project Settings")

    @app.get("/data", response_class=HTMLResponse)
    async def data_page(request: Request) -> HTMLResponse:
        return render(request, "data.html", current_page="data", page_title="Data and Codebook")

    @app.get("/split", response_class=HTMLResponse)
    async def split_page(request: Request) -> HTMLResponse:
        return render(request, "split.html", current_page="split", page_title="Split Builder")

    @app.get("/baseline", response_class=HTMLResponse)
    async def baseline_page(request: Request) -> HTMLResponse:
        return render(request, "baseline.html", current_page="baseline", page_title="Baseline Console")

    @app.get("/train", response_class=HTMLResponse)
    async def train_page(request: Request) -> HTMLResponse:
        return render(request, "train.html", current_page="train", page_title="Sweep Planner")

    @app.get("/runs", response_class=HTMLResponse)
    async def runs_page(request: Request) -> HTMLResponse:
        return render(request, "runs.html", current_page="runs", page_title="Run Ledger")

    return app


def create_app_from_env() -> FastAPI:
    return create_app(os.environ.get("CBMA_UI_PROJECT"))
