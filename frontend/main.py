import os
import base64
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
from fastapi.responses import HTMLResponse

app = FastAPI(title="UI Backend")
templates = Jinja2Templates(directory="templates")

AI_BASE = os.environ.get("AI_BACKEND_BASE", "http://localhost:8001")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...), conf_threshold: float = Form(0.25)):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    data = {"conf_threshold": str(conf_threshold)}
    try:
        r = requests.post(f"{AI_BASE}/detect", files=files, data=data, timeout=60)
        r.raise_for_status()
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e), "result": None})

    res = r.json()

    ann_path = res.get("annotated_image")
    ann_b64 = None
    if ann_path:
        try:
            img_resp = requests.get(f"{AI_BASE}{ann_path}")
            img_resp.raise_for_status()
            ann_b64 = base64.b64encode(img_resp.content).decode()
        except Exception:
            ann_b64 = None

    return templates.TemplateResponse("index.html", {"request": request, "result": res, "annotated_b64": ann_b64})
