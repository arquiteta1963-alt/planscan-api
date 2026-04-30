from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os
import json

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
    return {"status": "ok", "message": "PlanScan AI ativo"}

@app.post("/processar-planta")
async def processar_planta(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """
Analise esta planta baixa desenhada à mão.

Retorne SOMENTE JSON válido, sem markdown, sem explicação.

Formato obrigatório:
{
  "status": "success",
  "ambientes": [
    {"nome": "", "tipo": "", "observacao": ""}
  ],
  "portas": [
    {"local": "", "observacao": ""}
  ],
  "janelas": [
    {"local": "", "observacao": ""}
  ],
  "escadas": [
    {"local": "", "observacao": ""}
  ],
  "paredes": [
    {"descricao": "", "observacao": ""}
  ],
  "observacoes": [],
  "resumo": ""
}

Se algo estiver incerto, escreva "incerto".
Não invente medidas exatas.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    texto = response.output_text.strip()

    try:
        resultado = json.loads(texto)
    except Exception:
        resultado = {
            "status": "error",
            "raw": texto,
            "message": "A IA respondeu, mas não em JSON válido."
        }

    return resultado
