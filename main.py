from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import uuid

app = FastAPI(title="PlanScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "PlanScan API ativa com OpenCV + merge"}

def image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(thresh, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=80,
        maxLineGap=12,
    )

    detected = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.hypot(dx, dy)

            if length < 80:
                continue

            # manter apenas linhas quase horizontais ou verticais
            if dx > 12 and dy > 12:
                continue

            if dx >= dy:
                y = int((y1 + y2) / 2)
                x_start = min(x1, x2)
                x_end = max(x1, x2)
                detected.append({
                    "id": f"wall_{len(detected) + 1}",
                    "x1": int(x_start),
                    "y1": y,
                    "x2": int(x_end),
                    "y2": y,
                    "orientation": "horizontal",
                    "type": "wall",
                    "confidence": 0.85,
                })
            else:
                x = int((x1 + x2) / 2)
                y_start = min(y1, y2)
                y_end = max(y1, y2)
                detected.append({
                    "id": f"wall_{len(detected) + 1}",
                    "x1": x,
                    "y1": int(y_start),
                    "x2": x,
                    "y2": int(y_end),
                    "orientation": "vertical",
                    "type": "wall",
                    "confidence": 0.85,
                })

    return detected

def merge_lines(lines, threshold=20):
    merged = []

    for line in lines:
        added = False

        for m in merged:
            if line["orientation"] != m["orientation"]:
                continue

            if line["orientation"] == "horizontal":
                if abs(line["y1"] - m["y1"]) <= threshold:
                    if not (line["x2"] < m["x1"] - threshold or line["x1"] > m["x2"] + threshold):
                        m["x1"] = min(m["x1"], line["x1"])
                        m["x2"] = max(m["x2"], line["x2"])
                        m["y1"] = int((m["y1"] + line["y1"]) / 2)
                        m["y2"] = m["y1"]
                        added = True
                        break

            if line["orientation"] == "vertical":
                if abs(line["x1"] - m["x1"]) <= threshold:
                    if not (line["y2"] < m["y1"] - threshold or line["y1"] > m["y2"] + threshold):
                        m["y1"] = min(m["y1"], line["y1"])
                        m["y2"] = max(m["y2"], line["y2"])
                        m["x1"] = int((m["x1"] + line["x1"]) / 2)
                        m["x2"] = m["x1"]
                        added = True
                        break

        if not added:
            merged.append(line.copy())

    for i, wall in enumerate(merged):
        wall["id"] = f"wall_{i + 1}"

    return merged

def make_svg(width, height, walls):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>'
    ]

    for wall in walls:
        svg.append(
            f'<line x1="{wall["x1"]}" y1="{wall["y1"]}" '
            f'x2="{wall["x2"]}" y2="{wall["y2"]}" '
            f'stroke="black" stroke-width="3" stroke-linecap="round"/>'
        )

    svg.append("</svg>")
    return "".join(svg)

def make_preview(image, walls):
    preview = image.copy()

    for wall in walls:
        cv2.line(
            preview,
            (wall["x1"], wall["y1"]),
            (wall["x2"], wall["y2"]),
            (0, 0, 255),
            2
        )

    return preview

@app.post("/processar-planta")
async def processar_planta(file: UploadFile = File(...)):
    file_bytes = await file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "status": "error",
            "message": "Não foi possível ler a imagem enviada."
        }

    height, width = image.shape[:2]

    raw_lines = detect_lines(image)
    walls = merge_lines(raw_lines, threshold=20)

    svg = make_svg(width, height, walls)
    preview = make_preview(image, walls)

    return {
        "status": "success",
        "id": str(uuid.uuid4()),
        "width": width,
        "height": height,
        "rooms": [],
        "walls": walls,
        "doors": [],
        "windows": [],
        "uncertain": [],
        "summary": f"{len(raw_lines)} linhas brutas detectadas; {len(walls)} paredes após merge.",
        "svg": svg,
        "preview_base64": image_to_base64(preview),
        "unit": "cm",
        "scale": "1:50",
    }
