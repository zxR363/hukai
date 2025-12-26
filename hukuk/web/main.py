from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
import os
import sys

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
# from web.legal_engine import LegalSearchEngine, create_pdf_report_file
from web.mock_legal_engine import LegalSearchEngine, create_pdf_report_file

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

# ================== LOGIC ==================
async def run_search_task(search_id: str, req: SearchRequest):
    """Background task to run the search and stream logs explicitly to the specific client logic (broadcast for now)"""
    
    async def log_callback(msg: str):
        # We prefix logs with simple text. Frontend parses them.
        await manager.broadcast(f"LOG|{msg}")

    try:
        engine = LegalSearchEngine(log_callback=log_callback)
        neg_list = [w.strip().lower() for w in req.negatives.split(",")] if req.negatives else []
        
        advice, docs = await engine.run_analysis(req.story, req.topic, neg_list)
        
        # Generator PDF
        pdf_filename = f"Hukuki_Rapor_{search_id}.pdf"
        output_path = os.path.join("results", pdf_filename)
        
        if create_pdf_report_file(req.story, docs, advice, output_path):
             await manager.broadcast(f"PDF|{pdf_filename}")
        
        # Send final structured results as specialized JSON-like string
        # Or better yet, we can send a "DONE" event with data, but since we are simple:
        # We will send a large JSON payload via websocket for the UI to render results
        import json
        result_payload = {
            "advice": advice,
            "docs": docs
        }
        await manager.broadcast(f"RESULT|{json.dumps(result_payload)}")

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
    search_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_search_task, search_id, req)
    return {"status": "started", "search_id": search_id}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    return JSONResponse(status_code=404, content={"message": "File not found"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
