from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def lines_to_svg(lines, width, height):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    svg += '<g stroke="black" stroke-width="2" fill="none" stroke-linecap="round">'

    detected_walls = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" />'
            detected_walls.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "type": "wall_candidate"
            })

    svg += "</g></svg>"
    return svg, detected_walls

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "PlanScan API rodando com OpenCV"
    }

@app.post("/processar-planta")
async def processar_planta(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Imagem inválida"}
            )

        height, width = image.shape[:2]

        # Reduz imagem se for muito grande
        max_size = 1600
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            height, width = image.shape[:2]

        # Efeito scanner básico
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoise = cv2.GaussianBlur(gray, (5, 5), 0)
        scanned = cv2.adaptiveThreshold(
            denoise,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )

        # Detecção de linhas
        edges = cv2.Canny(scanned, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=60,
            maxLineGap=12
        )

        preview = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(preview, (x1, y1), (x2, y2), (0, 80, 255), 2)

        svg, detected_walls = lines_to_svg(lines, width, height)

        return {
            "id": str(uuid.uuid4()),
            "status": "success",
            "filename": file.filename,
            "width": width,
            "height": height,
            "preview_base64": encode_image_to_base64(preview),
            "scanned_base64": encode_image_to_base64(scanned),
            "svg": svg,
            "detected_elements": {
                "walls": detected_walls,
                "doors": [],
                "windows": [],
                "stairs": []
            },
            "message": "Imagem processada com sucesso"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
