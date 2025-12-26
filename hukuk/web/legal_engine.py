import sys
import os
import re
import uuid
import time
import shutil
import asyncio
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Dict, Optional

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_community.document_loaders import PyMuPDFLoader

# ================== AYARLAR ==================
SOURCES = {
    "mevzuat": {
        "folder": "mevzuatlar",
        "collection": "legal_statutes_v48",
        "desc": "MEVZUAT"
    },
    "emsal": {
        "folder": "belgeler",
        "collection": "legal_precedents_v48",
        "desc": "EMSAL KARAR"
    }
}

QDRANT_PATH = "qdrant_db_master"
EMBEDDING_MODEL = "nomic-embed-text"

SEARCH_LIMIT_PER_SOURCE = 30
SCORE_THRESHOLD = 0.40
LLM_RERANK_LIMIT = 15

# ==================================================
# 1️⃣ ARAÇLAR (HELPERS)
# ==================================================
def force_unlock_db():
    lock_file = os.path.join(QDRANT_PATH, ".lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except:
            pass

def extract_pdf_conclusion(file_path, char_limit=2500):
    try:
        if not os.path.exists(file_path): return "[Dosya bulunamadı.]"
        doc = fitz.open(file_path)
        total_pages = len(doc)
        text = ""
        start_page = max(0, total_pages - 2)
        for i in range(start_page, total_pages): text += doc[i].get_text()
        doc.close()
        return text[-char_limit:]
    except Exception as e:
        return f"[Karar okunamadı: {e}]"

def clean_text(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Note: multiprocessing functions must be at top level
def worker_embed_batch(args):
    texts, model_name = args
    embedder = OllamaEmbeddings(model=model_name)
    return embedder.embed_documents(texts)

# ==================================================
# 2️⃣ PDF REPORT CLASS
# ==================================================
class LegalReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'HUKUKI ANALIZ RAPORU', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', align='C')

def create_pdf_report_file(user_story, valid_docs, advice_text, output_path):
    pdf = LegalReport()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)

    def clean(t):
        if not t: return ""
        tr = {'ğ': 'g', 'ü': 'u', 'ş': 's', 'ı': 'i', 'ö': 'o', 'ç': 'c', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'İ': 'I',
              'Ö': 'O', 'Ç': 'C'}
        for k, v in tr.items(): t = t.replace(k, v)
        return t.encode('latin-1', 'replace').decode('latin-1')

    pdf.set_font(style='B', size=12);
    pdf.cell(0, 10, clean("1. OLAY:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(style='', size=10);
    pdf.multi_cell(w=pdf.epw, h=6, text=clean(user_story));
    pdf.ln(5)

    pdf.set_font(style='B', size=12);
    pdf.cell(0, 10, clean("2. KULLANILAN KAYNAKLAR:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for doc in valid_docs:
        # 1. Satır: Tür ve İsim
        pdf.set_font(style='B', size=9)
        source_title = f"[{doc['type']}] {doc['source']} (Sf. {doc['page']})"
        pdf.cell(0, 6, clean(source_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # 2. Satır: Rol (Alt satıra, girintili)
        pdf.set_font(style='B', size=8)
        pdf.cell(0, 5, clean(f"   Rol: {doc['role']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # 3. Satır: Sebep
        pdf.set_font(style='I', size=8)
        pdf.multi_cell(w=pdf.epw, h=4, text=clean(f"   Sebep: {doc['reason']}"));
        pdf.ln(2)

    pdf.add_page();
    pdf.set_font(style='B', size=12);
    pdf.cell(0, 10, clean("3. HUKUKI GORUS:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(style='', size=10);
    pdf.multi_cell(w=pdf.epw, h=6, text=clean(advice_text))
    try:
        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"PDF Error: {e}")
        return False

# ==================================================
# 3️⃣ ENGINE CLASS
# ==================================================
class LegalSearchEngine:
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self.log_callback = log_callback or (lambda x: print(x))
        self.log("🚀 ENGINE: Başlatılıyor...")
        
        force_unlock_db()
        try:
            self.client = QdrantClient(path=QDRANT_PATH)
            self.log("   ✅ Veritabanı bağlantısı BAŞARILI.")
        except Exception as e:
            self.log(f"\n❌ VERİTABANI HATASI: {e}")
            raise e

        self.llm = ChatOllama(model="qwen2.5", temperature=0.1)
        self.dense_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    def log(self, message: str):
        if self.log_callback:
            if asyncio.iscoroutinefunction(self.log_callback):
                # We can't await here easily if called from sync code, 
                # but we will handle async logging in the caller or wrapper
                pass
            else:
                self.log_callback(message)

    async def alog(self, message: str):
        """Async logger helper"""
        if self.log_callback:
            if asyncio.iscoroutinefunction(self.log_callback):
                await self.log_callback(message)
            else:
                self.log_callback(message)

    def generate_expanded_queries(self, story, topic):
        self.log("   ↳ 🧠 Sorgu Genişletiliyor...")
        try:
            prompt = f"GÖREV: Hukuki terimler.\nOLAY: {story}\nODAK: {topic}\n3 kısa cümle."
            res = self.llm.invoke(prompt).content
            return [line.strip() for line in res.splitlines() if len(line) > 5][:3]
        except:
            return [story]

    def assign_document_role(self, user_query, document_text):
        prompt = f"""
SEN HUKUKÇUSUN.
Sorgu: "{user_query}"
Belge: "{document_text[:800]}..."

GÖREV: Bu belge hukuki analizde nasıl kullanılmalı?
1. [DOĞRUDAN DELİL]: Belgedeki olay örgüsü ve maddi vakıalar, kullanıcının olayıyla birebir örtüşüyor.
2. [EMSAL İLKE]: Olay farklı olsa bile, belgedeki "Yargıtay İlkesi" veya "Hukuk Kuralı" konuya uygulanabilir.

SADECE ŞUNLARDAN BİRİNİ SEÇ:
[DOĞRUDAN DELİL] veya [EMSAL İLKE]
"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            if "DOĞRUDAN" in res: return "[DOĞRUDAN DELİL]"
            return "[EMSAL İLKE]"
        except:
            return "[EMSAL İLKE]"

    def check_relevance_judge_smart(self, user_query, negative_keywords, document_text):
        found_negative = None
        if negative_keywords:
            doc_lower = document_text.lower()
            for bad in negative_keywords:
                if re.search(rf"\b{re.escape(bad)}\b", doc_lower): found_negative = bad; break

        if found_negative:
            prompt = f"HUKUKÇU. Sorgu: '{user_query}'. Yasaklı: '{found_negative}'. Uygun mu? [RED]/[KABUL]."
            res = self.llm.invoke(prompt).content.strip()
            if "RED" in res: return False, f"⛔ YASAKLI: {res}"

        prompt_gen = f"""
SEN KIDEMLI BIR HUKUKCUSSUN.
SORGUNUN AMACI: Benzer Yargıtay içtihatlarını ve hukuki ilke kararlarını bulmak.
Sorgu: "{user_query}"
Belge: "{document_text[:700]}..."
SORU: Bu belge; hukuki ilke, yorum yaklaşımı, miras hukuku mantığı bakımından sorguyla ne derece BENZER?
SADECE BİRİNİ SEÇ: [ÇOK BENZER], [BENZER], [ZAYIF]
Altına tek cümlelik gerekçe yaz.
"""
        res = self.llm.invoke(prompt_gen).content.strip()
        is_ok = ("ÇOK BENZER" in res) or ("BENZER" in res)
        return is_ok, res

    async def run_analysis(self, story: str, topic: str, negatives: List[str]):
        """
        Main entry point for web search.
        """
        await self.alog("-" * 60)
        await self.alog(f"📝 Olay: {story}")
        await self.alog(f"🎯 Odak: {topic}")
        
        expanded = self.generate_expanded_queries(story, topic)
        full_query = f"{story} {topic} " + " ".join(expanded)
        await self.alog(f"   ✓ Sorgu Genişletildi: {len(full_query)} karakter")

        await self.alog("\n🔍 Belgeler Taranıyor (Dual Search - Aşama 1)...")
        # Embedding can be blocking, so run in executor if needed, but for now direct call
        query_vector = self.dense_embedder.embed_query(full_query)
        
        all_candidates = []
        for key, config in SOURCES.items():
            results = self.client.query_points(collection_name=config["collection"], query=query_vector, limit=40).points
            for hit in results:
                if 'type' not in hit.payload: hit.payload['type'] = config['desc']
                all_candidates.append(hit)

        unique_docs = {}
        for hit in all_candidates:
            if hit.score < SCORE_THRESHOLD: continue
            key = f"{hit.payload['source']}_{hit.payload['page']}"
            if key not in unique_docs or hit.score > unique_docs[key].score: unique_docs[key] = hit

        candidates = sorted(unique_docs.values(), key=lambda x: x.score, reverse=True)[:LLM_RERANK_LIMIT]
        
        if not candidates:
            await self.alog("🔴 Uygun belge bulunamadı.")
            return None, []

        await self.alog("\n⚖️  Akıllı Yargıç Değerlendiriyor (Aşama 2: Rol Atama)...")
        valid_docs = []
        
        for hit in candidates:
            doc_text = hit.payload['page_content']
            source = hit.payload['source']
            page = hit.payload['page']
            type_desc = hit.payload['type']

            is_ok, reason = self.check_relevance_judge_smart(story, negatives, doc_text)
            norm_score = min(max(hit.score, 0), 1) * 100

            if is_ok:
                role = self.assign_document_role(story, doc_text)
                await self.alog(f"✅ [{type_desc}] {source} | Güven: %{norm_score:.1f} | Rol: {role}")

                extra_context = ""
                if type_desc == "EMSAL KARAR":
                    real_path = os.path.join(SOURCES["emsal"]["folder"], source)
                    verdict = extract_pdf_conclusion(real_path)
                    extra_context = f"\n\n🛑 [OTOMATİK EKLENEN KARAR SONUCU ({source})]:\n{verdict}\n🛑 KARAR SONU."

                valid_docs.append({
                    "source": source, "page": page, "type": type_desc, "role": role,
                    "text": doc_text + extra_context, "score": norm_score, "reason": reason
                })
            else:
                await self.alog(f"❌ [{type_desc}] {source} | Güven: %{norm_score:.1f} | Red: {reason[:50]}...")

        if not valid_docs:
            await self.alog("🔴 Yargıç tüm adayları eledi.")
            return None, []

        context_str = ""
        for i, d in enumerate(valid_docs):
            context_str += f""">>> BELGE #{i + 1}\nTÜR: [{d['type']}]\nROL: {d['role']}\nDOSYA ADI: {d['source']}\nSKOR: %{d['score']:.1f}\nİÇERİK:\n{d['text']}\n=========================================\n"""

        await self.alog("\n🧑‍⚖️  AVUKAT YAZIYOR (Role-Aware Mode)...")
        
        system_content = """SEN KIDEMLİ BİR HUKUKÇUSUN.

⚠️ BELGE KULLANIM KURALLARI (ROLLER):

1. **[DOĞRUDAN DELİL] Etiketli Belgeler:**
   - Olay örgüsünü kullanıcının olayıyla karşılaştır.
   - "Benzer olayda Yargıtay şöyle demiştir..." de.

2. **[EMSAL İLKE] Etiketli Belgeler (ÇOK ÖNEMLİ):**
   - Olay örgüsünü (boşanma, trafik vb.) ASLA ANLATMA.
   - Sadece "Yargıtay İlkesini" veya "Hukuk Kuralını" al.
   - "Yargıtay'ın yerleşik içtihadına göre..." de.

3. **GENEL KURALLAR:**
   - Belge #1'den başla, sırayla git.
   - Kaynak ismini (Dosya Adı) birebir kullan.

FORMAT:
A. MEVZUAT DAYANAKLARI
B. İLGİLİ EMSAL KARARLAR (Rollerine göre ayırarak yaz)
C. SONUÇ VE TAVSİYE"""

        user_content = f"""Aşağıdaki "DELİLLER" listesinde sunulan belgeleri kullanarak olayı analiz et.
OLAY: "{story}"
ODAK: "{topic}"

DELİLLER:
{context_str}

ANALİZİ BAŞLAT:"""

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]
        
        full_res = ""
        # Stream response
        for chunk in self.llm.stream(messages):
            c = chunk.content
            full_res += c
            # For web, we might want to stream this too, but for now just accumulate
            # Or emit special events for streaming text
            # await self.alog(f"STREAM:{c}") # Optional logic for future
        
        await self.alog("\n✅ Analiz Tamamlandı.")
        return full_res, valid_docs
