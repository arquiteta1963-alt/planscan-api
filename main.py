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
    return {"status": "ok", "message": "PlanScan API ativa com OpenCV"}


def image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # melhora contraste
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # detecta bordas
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # detecta linhas
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=45,
        maxLineGap=12,
    )

    detected = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            if length < 40:
                continue

            detected.append({
                "id": f"wall_{len(detected) + 1}",
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "type": "detected_line",
                "confidence": 0.65
            })

    return detected


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

    walls = detect_lines(image)
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
        "summary": f"{len(walls)} linhas detectadas por OpenCV.",
        "svg": svg,
        "preview_base64": image_to_base64(preview),
        "unit": "cm",
        "scale": "1:50"
    }
