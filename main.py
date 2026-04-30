from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os
import json
import re

app = FastAPI(title="PlanScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def home():
    return {"status": "ok", "message": "PlanScan API ativa"}


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return json.loads(text.strip())


def make_svg(data):
    rooms = data.get("rooms", [])
    walls = data.get("walls", [])
    doors = data.get("doors", [])
    windows = data.get("windows", [])

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">',
        '<rect width="100%" height="100%" fill="white"/>'
    ]

    for wall in walls:
        x1 = wall.get("x1", 0)
        y1 = wall.get("y1", 0)
        x2 = wall.get("x2", 0)
        y2 = wall.get("y2", 0)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="6" stroke-linecap="square"/>')

    for door in doors:
        x = door.get("x", 0)
        y = door.get("y", 0)
        w = door.get("width", 45)
        svg.append(f'<path d="M{x},{y} A{w},{w} 0 0 1 {x+w},{y+w}" fill="none" stroke="black" stroke-width="3"/>')

    for window in windows:
        x1 = window.get("x1", 0)
        y1 = window.get("y1", 0)
        x2 = window.get("x2", 0)
        y2 = window.get("y2", 0)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2563eb" stroke-width="4"/>')

    for room in rooms:
        name = room.get("name", "Ambiente")
        x = room.get("label_x", 100)
        y = room.get("label_y", 100)
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="22" text-anchor="middle" fill="#111">{name}</text>')

    svg.append("</svg>")
    return "".join(svg)


@app.post("/processar-planta")
async def processar_planta(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
Você é uma IA especializada em leitura de plantas baixas arquitetônicas desenhadas à mão.

Analise a imagem e retorne SOMENTE JSON válido, sem markdown e sem explicações.

Objetivo:
Criar uma primeira versão vetorial editável da planta.

Retorne obrigatoriamente neste formato:

{
  "status": "success",
  "summary": "",
  "rooms": [
    {
      "id": "room_1",
      "name": "Bedroom",
      "type": "quarto",
      "label_x": 200,
      "label_y": 200,
      "confidence": 0.8
    }
  ],
  "walls": [
    {
      "id": "wall_1",
      "x1": 100,
      "y1": 100,
      "x2": 500,
      "y2": 100,
      "type": "external",
      "confidence": 0.8
    }
  ],
  "doors": [
    {
      "id": "door_1",
      "x": 200,
      "y": 300,
      "width": 45,
      "swing": "left",
      "confidence": 0.7
    }
  ],
  "windows": [
    {
      "id": "window_1",
      "x1": 300,
      "y1": 100,
      "x2": 380,
      "y2": 100,
      "confidence": 0.7
    }
  ],
  "uncertain": [],
  "scale": "1:50",
  "unit": "m"
}

Regras:
- Use coordenadas aproximadas em um canvas 1000x700.
- Não invente medidas reais.
- Se algo estiver incerto, coloque em "uncertain".
- Priorize paredes externas, divisórias principais, portas, janelas e nomes dos cômodos.
- As coordenadas precisam permitir gerar um SVG visual aproximado.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    raw_text = response.output_text.strip()

    try:
        data = extract_json(raw_text)
    except Exception:
        return {
            "status": "error",
            "message": "A IA respondeu, mas não retornou JSON válido.",
            "raw": raw_text
        }

    if data.get("status") != "success":
        data["status"] = "success"

    data["svg"] = make_svg(data)

    return data
