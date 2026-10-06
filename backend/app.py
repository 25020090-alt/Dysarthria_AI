from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
from backend.model_service import ai_service

app = FastAPI(title="Dysarthria AI Server")

# Gắn giao diện web vào đường dẫn gốc (/web)
app.mount("/web", StaticFiles(directory="frontend", html=True), name="frontend")

@app.post("/api/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    try:
        audio_bytes = await audio_file.read()
        result_text = ai_service.transcribe_audio(audio_bytes)
        
        return JSONResponse(content={
            "success": True,
            "text": result_text
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    print("\n[SERVER] Máy chủ đang chạy tại: http://localhost:8000/web")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
