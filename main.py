from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import uuid
import math

app = FastAPI(title="PlanScan API - Geometry Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "PlanScan API ativa - OpenCV + Geometry Engine"}

def image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    return clean

def detect_raw_lines(binary):
    edges = cv2.Canny(binary, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=70,
        maxLineGap=18
    )

    result = []

    if lines is None:
        return result

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length < 70:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))

        # Snap horizontal / vertical
        if angle < 12 or angle > 168:
            y = int((y1 + y2) / 2)
            result.append({
                "x1": int(min(x1, x2)),
                "y1": y,
                "x2": int(max(x1, x2)),
                "y2": y,
                "orientation": "horizontal",
                "type": "wall"
            })

        elif 78 < angle < 102:
            x = int((x1 + x2) / 2)
            result.append({
                "x1": x,
                "y1": int(min(y1, y2)),
                "x2": x,
                "y2": int(max(y1, y2)),
                "orientation": "vertical",
                "type": "wall"
            })

    return result

def merge_lines(lines, axis_threshold=18, gap_threshold=45):
    horizontals = [l for l in lines if l["orientation"] == "horizontal"]
    verticals = [l for l in lines if l["orientation"] == "vertical"]

    def merge_group(group, orientation):
        if not group:
            return []

        if orientation == "horizontal":
            group.sort(key=lambda l: (l["y1"], l["x1"]))
        else:
            group.sort(key=lambda l: (l["x1"], l["y1"]))

        merged = []

        for line in group:
            added = False

            for m in merged:
                if orientation == "horizontal":
                    same_axis = abs(line["y1"] - m["y1"]) <= axis_threshold
                    overlapping = not (
                        line["x1"] > m["x2"] + gap_threshold or
                        line["x2"] < m["x1"] - gap_threshold
                    )

                    if same_axis and overlapping:
                        m["x1"] = min(m["x1"], line["x1"])
                        m["x2"] = max(m["x2"], line["x2"])
                        m["y1"] = int((m["y1"] + line["y1"]) / 2)
                        m["y2"] = m["y1"]
                        added = True
                        break

                else:
                    same_axis = abs(line["x1"] - m["x1"]) <= axis_threshold
                    overlapping = not (
                        line["y1"] > m["y2"] + gap_threshold or
                        line["y2"] < m["y1"] - gap_threshold
                    )

                    if same_axis and overlapping:
                        m["y1"] = min(m["y1"], line["y1"])
                        m["y2"] = max(m["y2"], line["y2"])
                        m["x1"] = int((m["x1"] + line["x1"]) / 2)
                        m["x2"] = m["x1"]
                        added = True
                        break

            if not added:
                merged.append(line.copy())

        return merged

    merged = merge_group(horizontals, "horizontal") + merge_group(verticals, "vertical")

    cleaned = []
    for line in merged:
        length = abs(line["x2"] - line["x1"]) if line["orientation"] == "horizontal" else abs(line["y2"] - line["y1"])
        if length >= 60:
            cleaned.append(line)

    for i, line in enumerate(cleaned):
        line["id"] = f"wall_{i+1}"
        line["confidence"] = 0.9

    return cleaned

def build_svg(width, height, walls):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>'
    ]

    for wall in walls:
        svg.append(
            f'<line x1="{wall["x1"]}" y1="{wall["y1"]}" x2="{wall["x2"]}" y2="{wall["y2"]}" '
            f'stroke="black" stroke-width="5" stroke-linecap="square"/>'
        )

    svg.append("</svg>")
    return "".join(svg)

def build_preview(image, walls):
    preview = image.copy()

    for wall in walls:
        cv2.line(
            preview,
            (wall["x1"], wall["y1"]),
            (wall["x2"], wall["y2"]),
            (0, 0, 255),
            3
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
            "message": "Imagem inválida."
        }

    height, width = image.shape[:2]

    binary = preprocess_image(image)
    raw_lines = detect_raw_lines(binary)
    walls = merge_lines(raw_lines)

    svg = build_svg(width, height, walls)
    preview = build_preview(image, walls)

    return {
        "status": "success",
        "id": str(uuid.uuid4()),
        "width": width,
        "height": height,
        "raw_count": len(raw_lines),
        "wall_count": len(walls),
        "walls": walls,
        "doors": [],
        "windows": [],
        "rooms": [],
        "svg": svg,
        "preview_base64": image_to_base64(preview),
        "summary": f"{len(raw_lines)} linhas detectadas; {len(walls)} paredes consolidadas.",
        "engine": "opencv_geometry_v1"
    }
