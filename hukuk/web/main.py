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
    from web.legal_engine import LegalSearchEngine, LegalJudge, LegalReporter
except ImportError:
    from legal_engine import LegalSearchEngine, LegalJudge, LegalReporter

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
        self.active_connections: dict = {} # clientId -> WebSocket
        self.active_tasks: dict = {} # clientId -> asyncio.Task

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Kill the task if it's still running for this client
        if client_id in self.active_tasks:
            task = self.active_tasks[client_id]
            if not task.done():
                task.cancel()
                print(f"DEBUG: Task for client {client_id} cancelled.")
            del self.active_tasks[client_id]

    async def broadcast(self, message: str, target_id: str = None):
        if target_id and target_id in self.active_connections:
            try: await self.active_connections[target_id].send_text(message)
            except: pass
        else:
            for conn in list(self.active_connections.values()):
                try: await conn.send_text(message)
                except: pass

manager = ConnectionManager()
engine_lock = asyncio.Lock()

# ================== MODELS ==================
class SearchRequest(BaseModel):
    story: str
    topic: str
    negatives: str = ""
    clientId: Optional[str] = None

class EvaluateRequest(BaseModel):
    story: str
    topic: str
    negatives: str = ""
    candidates: List[dict]
    clientId: Optional[str] = None

# ================== LOGIC ==================
# ================== LOGIC ==================
async def run_search_task(req: SearchRequest):
    """
    TAM PİPELİNE - AŞAMA 1 & 2 (Arama + Yargılama)
    Mantık: legal_engine.py L471-L502
    """
    try:
        await manager.broadcast("LOG|🚀 Analiz başlatılıyor...", target_id=req.clientId)
        
        # 1. Motor Hazırlığı
        engine = LegalSearchEngine()
        judge = LegalJudge()
        
        # 2. Veritabanı Bağlantısı (L108-L113)
        await manager.broadcast("LOG|🔌 Veritabanı bağlantısı kontrol ediliyor...", target_id=req.clientId)
        is_connected = await asyncio.to_thread(engine.connect_db)
        if not is_connected:
            await manager.broadcast("LOG|❌ HATA: Veritabanına bağlanılamadı. Kilit dosyası veya path sorunu olabilir.", target_id=req.clientId)
            return
        await manager.broadcast("LOG|✅ Veritabanı bağlantısı başarılı.", target_id=req.clientId)

        # 2. GİRDİ DENETÇİSİ (L115-L118)
        await manager.broadcast("LOG|🔍 Kullanıcı girdisi kontrol ediliyor...", target_id=req.clientId)
        is_valid = await asyncio.to_thread(engine.validate_user_input, req.story, req.topic)
        if not is_valid:
            await manager.broadcast("LOG|❌ HATA: Kullanıcı girdisi geçersiz.", target_id=req.clientId)
            return
        await manager.broadcast("LOG|✅ Kullanıcı girdisi geçerli.", target_id=req.clientId)

        # 3. SORGÜ GENİŞLETME (L478-L479)
        await manager.broadcast("LOG|🧠 Hukuki terimler genişletiliyor...", target_id=req.clientId)
        expanded = await asyncio.to_thread(judge.generate_expanded_queries, req.story, req.topic)
        full_query = f"{req.story} {req.topic} " + " ".join(expanded)
        await manager.broadcast(f"LOG|✓ Sorgu hazırlandı: {len(full_query)} karakter.", target_id=req.clientId)
        
        # 4. ARAMA (L483)
        await manager.broadcast("LOG|🔍 Veritabanında eşleşen belgeler taranıyor...", target_id=req.clientId)
        candidates = await asyncio.to_thread(engine.retrieve_raw_candidates, full_query)
        
        if not candidates:
            await manager.broadcast("LOG|🔴 Arama sonucu: Uygun benzerlikte belge bulunamadı.", target_id=req.clientId)
            return
        
        # 5. YARGILAMA / FİLTRELEME (L487-L488)
        await manager.broadcast("LOG|⚖️ Akıllı Yargıç belgeleri analiz ediyor...", target_id=req.clientId)
        neg_list = [w.strip().lower() for w in req.negatives.split(",")] if req.negatives else []
        valid_docs = await asyncio.to_thread(judge.evaluate_candidates, candidates, req.story, req.topic, neg_list)
        
        if not valid_docs:
            await manager.broadcast("LOG|🔴 Yargıç analizi: Mevcut belgelerin hiçbiri kriterlere uygun bulunmadı.", target_id=req.clientId)
            return

        # Adayları UI listesine gönder
        ui_candidates = []
        for c in valid_docs:
            ui_candidates.append({
                "source": c['source'],
                "page": c.get('page', 0),
                "type": c['type'],
                "page_content": c.get('text', ''), # valid_docs uses 'text' field
                "score": c['score'], # Alreadly normalized in legal_engine
                "role": c.get('role', '[EMSAL İLKE]'),
                "reason": c.get('reason', '')
            })
        await manager.broadcast(f"SEARCH_RESULT|{json.dumps(ui_candidates)}", target_id=req.clientId)

        await manager.broadcast(f"LOG|✅ {len(valid_docs)} adet kesin uyumlu belge seçildi. Analiz raporu yazılabilir.", target_id=req.clientId)

    except asyncio.CancelledError:
        print("DEBUG: Task was cancelled by user (refresh/disconnect).")

    except Exception as e:
        await manager.broadcast(f"ERROR|Pipeline Hatası: {str(e)}", target_id=req.clientId)
        print(f"Pipeline Error: {e}")
    finally:
        try: await asyncio.to_thread(engine.close)
        except: pass
        await manager.broadcast("STATUS|READY")

async def run_evaluation_task(search_id: str, req: EvaluateRequest):
    """
    TAM PİPELİNE - AŞAMA 3 (Yazma + Raporlama)
    Mantık: legal_engine.py L503-L506
    """
    try:
        await manager.broadcast("LOG|📝 Hukuki görüş yazımı başlatılıyor...", target_id=req.clientId)
        judge = LegalJudge()
        reporter = LegalReporter()
        
        valid_docs = req.candidates
        
        # Context oluştur (L491-L493)
        context_str = ""
        for i, d in enumerate(valid_docs):
            context_str += f""">>> BELGE #{i + 1}\nTÜR: [{d['type']}]\nROL: {d.get('role', '[EMSAL İLKE]')}\nDOSYA ADI: {d['source']}\nSKOR: %{d['score']:.1f}\nİÇERİK:\n{d.get('page_content') or d.get('text', '')}\n=========================================\n"""

        # 5. YAZMA (L505)
        await manager.broadcast("LOG|🧑‍⚖️ Avukat analizini hazırlıyor (bu işlem 1-2 dakika sürebilir)...", target_id=req.clientId)
        full_advice = await asyncio.to_thread(judge.generate_final_opinion, req.story, req.topic, context_str)

        # 6. RAPORLAMA (L506)
        await manager.broadcast("LOG|📄 PDF rapor dosyası oluşturuluyor...", target_id=req.clientId)
        pdf_filename = f"Hukuki_Rapor_{search_id}.pdf"
        output_path = os.path.join("results", pdf_filename)
        os.makedirs("results", exist_ok=True)
        
        await asyncio.to_thread(reporter.create_report, req.story, valid_docs, full_advice, output_path)
        await manager.broadcast(f"PDF|{pdf_filename}", target_id=req.clientId)

        # Sonuçları UI'a gönder
        result_payload = {
            "advice": full_advice,
            "docs": valid_docs
        }
        await manager.broadcast(f"RESULT|{json.dumps(result_payload)}", target_id=req.clientId)
        await manager.broadcast("LOG|🏁 Analiz ve raporlama başarıyla tamamlandı. Raporu indirebilirsiniz.", target_id=req.clientId)

    except asyncio.CancelledError:
         print("DEBUG: Evaluation task cancelled.")
    except Exception as e:
        await manager.broadcast(f"ERROR|Değerlendirme Hatası: {str(e)}", target_id=req.clientId)
        print(f"Evaluation Error: {e}")

# ================== ENDPOINTS ==================
@app.get("/")
async def get():
    return FileResponse('web/static/index.html')

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket, clientId: str = None):
    if not clientId:
        clientId = "anonymous"
    await manager.connect(websocket, clientId)
    
    # Broadcast initial system status to this client
    if engine_lock.locked():
        await websocket.send_text("STATUS|BUSY")
    else:
        await websocket.send_text("STATUS|READY")
        
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(clientId)

@app.post("/api/search")
async def start_search(req: SearchRequest):
    # Wrap in a locked task
    async def locked_search():
        async with engine_lock:
            await manager.broadcast("STATUS|BUSY")
            await run_search_task(req)
            
    task = asyncio.create_task(locked_search())
    if req.clientId:
        manager.active_tasks[req.clientId] = task
    return {"status": "started"}

@app.post("/api/evaluate")
async def start_evaluate(req: EvaluateRequest):
    search_id = str(uuid.uuid4())[:8]
    # No engine lock needed for evaluation (Pure LLM/PDF)
    task = asyncio.create_task(run_evaluation_task(search_id, req))
    if req.clientId:
        manager.active_tasks[req.clientId] = task
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
