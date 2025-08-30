import os
import io
import uuid
import json
import urllib.request
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="AI Backend - Object Detection")

MODEL_PATH = "models/yolov3-tiny.pt"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv3-tiny weights...")
    url = "https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3-tiny.pt"
    urllib.request.urlretrieve(url, MODEL_PATH)

model = YOLO(MODEL_PATH) 
class_names = model.names  


font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# font_path = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
font_size = 30

try:
    font = ImageFont.truetype(font_path, font_size)
except Exception:
    print("loading default font")
    font = ImageFont.load_default()

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = Form(0.25)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload image file only.")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Cannot open image") from e

    uid = uuid.uuid4().hex
    in_path = f"temp_{uid}.jpg"
    img.save(in_path)

    results = model.predict(source=in_path, device="cpu", conf=conf_threshold, verbose=False)
    res = results[0]

    boxes = getattr(res, "boxes", None)
    preds = []
    if boxes is not None and len(boxes) > 0:
        try:
            xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            xyxy = boxes.xyxy
        try:
            confs = boxes.conf.cpu().numpy()
        except Exception:
            confs = boxes.conf
        try:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cls_ids = boxes.cls

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [float(x) for x in box]
            cid = int(cls_ids[i])
            preds.append({
                "class_id": cid,
                "class_name": class_names.get(cid, str(cid)),
                "confidence": float(confs[i]),
                "bbox": [x1, y1, x2, y2]
            })

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    for p in preds:
        x1, y1, x2, y2 = p["bbox"]
        label = f'{p["class_name"]} {p["confidence"]:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        if font:
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(label, font=font)

            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1])
            draw.text((x1, y1 - text_h), label, fill="red", font=font)
            
        else:
            print("dasjkfhkjasf")
            draw.text((x1, y1), label, fill="red")

    out_img_name = f"{uid}_annotated.jpg"
    out_json_name = f"{uid}.json"
    out_img_path = os.path.join("outputs", out_img_name)
    out_json_path = os.path.join("outputs", out_json_name)

    draw_img.save(out_img_path)
    with open(out_json_path, "w") as f:
        json.dump({
            "original_filename": file.filename,
            "width": img.width,
            "height": img.height,
            "predictions": preds
        }, f, indent=2)

    try:
        os.remove(in_path)
    except Exception:
        pass

    return JSONResponse({
        "original_filename": file.filename,
        "predictions": preds,
        "annotated_image": f"/outputs/{out_img_name}",
        "json_file": f"/outputs/{out_json_name}"
    })


@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = os.path.join("outputs", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="file not found")

    if filename.lower().endswith(".json"):
        return FileResponse(path, media_type="application/json", filename=filename)
    else:
        return FileResponse(path, media_type="image/jpeg", filename=filename)
