from fastapi import FastAPI, UploadFile, File
import base64
import os
from openai import OpenAI

app = FastAPI()

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
                    {"type": "input_text", "text": "Analise esta planta baixa e retorne um JSON estruturado com: paredes, portas, cômodos e layout."},
                    {
                        {
    "type": "input_image",
    "image_url": f"data:image/jpeg;base64,{image_base64}"
}
                    }
                ]
            }
        ]
    )

 return {"resultado": response.output_text}
