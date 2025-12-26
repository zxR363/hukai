from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
import os
import sys
import json
from datetime import datetime

# ==================# --- ANALYSIS PERSISTENCE ---
class AnalysisData(BaseModel):
    name: str
    dashboardState: dict
    reportData: list
    judgeMetadata: dict
    timestamp: str

# ================== PATH FIXER ==================
# Otomatik olarak çalışma dizinini proje kök dizinine (hukuk) ayarlar.
# Bu sayede PyCharm'dan veya terminalden nereden çalıştırırsanız çalıştırın hatasız çalışır.
current_file_path = os.path.abspath(__file__) # .../hukuk/web/main.py
web_dir = os.path.dirname(current_file_path) # .../hukuk/web
project_root = os.path.dirname(web_dir)      # .../hukuk

# 1. Proje kökünü Python yoluna ekle (ModuleNotFoundError çözümü)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Çalışma dizinini proje kökü yap (Directory not found çözümü)
os.chdir(project_root)

import uuid
try:
    from web.legal_engine import LegalSearchEngine, create_pdf_report_file
except ImportError:
    from legal_engine import LegalSearchEngine, create_pdf_report_file
# from web.mock_legal_engine import LegalSearchEngine, create_pdf_report_file

app = FastAPI(title="Legal Suite V55 Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# ================== WEBSOCKET MANAGER ==================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# ================== MODELS ==================
class SearchRequest(BaseModel):
    story: str
    topic: str
    negatives: str = ""

class EvaluateRequest(BaseModel):
    story: str
    topic: str
    negatives: str = ""
    candidates: List[dict]

# ================== LOGIC ==================
async def run_search_task(req: SearchRequest):
    """Sadece belgeleri arar ve aday listesini döner."""
    async def log_callback(msg: str):
        await manager.broadcast(f"LOG|{msg}")

    try:
        engine = LegalSearchEngine(log_callback=log_callback)
        docs = await engine.search_docs(req.story, req.topic)
        
        # Arama bittiğinde adayları UI'a gönder
        import json
        await manager.broadcast(f"SEARCH_RESULT|{json.dumps(docs)}")

    except Exception as e:
        await manager.broadcast(f"ERROR|{str(e)}")

async def run_evaluation_task(search_id: str, req: EvaluateRequest):
    """Seçili belgeleri değerlendirir ve analiz yazar."""
    async def log_callback(msg: str):
        await manager.broadcast(f"LOG|{msg}")

    try:
        engine = LegalSearchEngine(log_callback=log_callback)
        neg_list = [w.strip().lower() for w in req.negatives.split(",")] if req.negatives else []
        
        advice, docs = await engine.evaluate_documents(req.story, req.topic, neg_list, req.candidates)
        
        if advice and docs:
            # Generator PDF
            pdf_filename = f"Hukuki_Rapor_{search_id}.pdf"
            output_path = os.path.join("results", pdf_filename)
            
            if create_pdf_report_file(req.story, docs, advice, output_path):
                 await manager.broadcast(f"PDF|{pdf_filename}")
            
            import json
            result_payload = {
                "advice": advice,
                "docs": docs
            }
            await manager.broadcast(f"RESULT|{json.dumps(result_payload)}")
        else:
             await manager.broadcast(f"LOG|🔴 Değerlendirme tamamlanamadı (Uygun belge kalmadı).")

    except Exception as e:
        await manager.broadcast(f"ERROR|{str(e)}")

# ================== ENDPOINTS ==================
@app.get("/")
async def get():
    return FileResponse('web/static/index.html')

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # simple ping-pong or command handling if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/search")
async def start_search(req: SearchRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_search_task, req)
    return {"status": "started"}

@app.post("/api/evaluate")
async def start_evaluate(req: EvaluateRequest, background_tasks: BackgroundTasks):
    search_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_evaluation_task, search_id, req)
    return {"status": "started", "search_id": search_id}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return JSONResponse(status_code=404, content={"message": "File not found"})

# --- ANALYSIS PERSISTENCE ---
class AnalysisData(BaseModel):
    name: str
    dashboardState: dict
    reportData: list
    judgeMetadata: dict
    timestamp: str

@app.post("/api/save_analysis")
async def save_analysis(data: AnalysisData):
    """Saves the current analysis state to a JSON file."""
    try:
        # Create DB dir if not exists
        db_dir = os.path.join(project_root, "web", "analysis_db")
        os.makedirs(db_dir, exist_ok=True)

        # Generate filename: timestamp_slug.json
        safe_name = "".join([c for c in data.name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.json"
        
        file_path = os.path.join(db_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data.dict(), f, ensure_ascii=False, indent=4)
            
        return {"status": "success", "message": "Analiz başarıyla kaydedildi.", "filename": filename}
    except Exception as e:
        print(f"Save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete_analysis/{filename}")
async def delete_analysis(filename: str):
    """Deletes a specific analysis file."""
    try:
        db_dir = os.path.join(project_root, "web", "analysis_db")
        file_path = os.path.join(db_dir, filename)
        
        if not os.path.exists(file_path):
             print(f"Delete Failed: File not found at {file_path}")
             raise HTTPException(status_code=404, detail="Analiz dosyası bulunamadı.")

        print(f"Deleting analysis file: {file_path}")
        os.remove(file_path)
        return {"status": "success", "message": "Analiz başarıyla silindi."}
    except Exception as e:
        err_msg = f"Silme hatası: {str(e)}"
        print(f"Delete error for {filename}: {err_msg}")
        raise HTTPException(status_code=500, detail=err_msg)

@app.get("/api/history")
async def get_history():
    """Lists all saved analyses."""
    try:
        db_dir = os.path.join(project_root, "web", "analysis_db")
        if not os.path.exists(db_dir):
            return []
            
        files = []
        for f in os.listdir(db_dir):
            if f.endswith(".json"):
                path = os.path.join(db_dir, f)
                stats = os.stat(path)
                
                try:
                    with open(path, "r", encoding="utf-8") as json_file:
                        content = json.load(json_file)
                        files.append({
                            "filename": f,
                            "name": content.get("name", "İsimsiz Analiz"),
                            "timestamp": content.get("timestamp"),
                            "doc_count": len(content.get("reportData", [])),
                            "file_size": stats.st_size
                        })
                except Exception:
                    continue
        
        # Sort by timestamp descending
        files.sort(key=lambda x: x["filename"], reverse=True)
        return files
    except Exception as e:
        print(f"History error: {e}")
        return []


@app.get("/api/load_analysis/{filename}")
async def load_analysis(filename: str):
    """Loads a specific analysis file."""
    try:
        db_dir = os.path.join(project_root, "web", "analysis_db")
        file_path = os.path.join(db_dir, filename)
        
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail="Analiz dosyası bulunamadı.")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data
    except Exception as e:
        print(f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
