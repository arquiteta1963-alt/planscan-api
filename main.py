from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os

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
Analise esta imagem de planta baixa desenhada à mão.

Retorne APENAS um JSON válido com:
{
  "ambientes": [],
  "portas": [],
  "janelas": [],
  "escadas": [],
  "paredes": [],
  "observacoes": [],
  "resumo": ""
}

Identifique cômodos, portas, escadas, paredes principais, janelas e problemas visíveis.
Não invente medidas exatas. Se algo estiver incerto, marque como "incerto".
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

    return {"resultado": response.output_text}
