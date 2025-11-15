from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, traceback
from model_utils import analyze_image_pil, load_age_model

app = FastAPI(title="Aging Phenotype Analyzer")

# CORS: permitir todas las fuentes para facilitar pruebas. En producción, restringe a tu dominio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Intentará cargar model_weights.pth si lo colocas en la misma carpeta (opcional)
age_model = load_age_model(model_path="model_weights.pth")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = analyze_image_pil(img, age_model=age_model)
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Para ejecutar localmente (solo si arrancas con python main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
