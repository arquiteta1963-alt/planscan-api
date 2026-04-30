from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

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
    return {
        "status": "ok",
        "message": "PlanScan API rodando"
    }

@app.post("/processar-planta")
async def processar_planta(file: UploadFile = File(...)):
    return {
        "status": "recebido",
        "filename": file.filename,
        "message": "Imagem recebida com sucesso"
    }
