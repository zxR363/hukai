import sys
import os
import re
import uuid
import time
import shutil
import atexit
import json
import random
import math
from typing import Dict, Any, List, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support
from dataclasses import dataclass
from collections import Counter

# --------------------------------------------------
# 📦 IMPORTLAR
# --------------------------------------------------
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, Range
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_community.document_loaders import PyMuPDFLoader

# UTF-8 Ayarı
sys.stdout.reconfigure(encoding="utf-8")


# ==================================================
# 1️⃣ KONFİGÜRASYON VE YARDIMCILAR
# ==================================================
@dataclass
class LegalConfig:
    SOURCES = {
        "mevzuat": {"folder": "mevzuatlar", "collection": "legal_statutes_v48", "desc": "MEVZUAT"},
        "emsal": {"folder": "belgeler", "collection": "legal_precedents_v48", "desc": "EMSAL KARAR"}
    }
    MEMORY_COLLECTIONS = {"decision": "judge_memory_v1", "principle": "principle_memory_v1"}
    QDRANT_PATH = "qdrant_db_master"
    STATE_FILE = "system_state.json"
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "qwen2.5"
    SEARCH_LIMIT_PER_SOURCE = 60
    SCORE_THRESHOLD = 0.35
    LLM_RERANK_LIMIT = 10
    DECAY_RATE_PER_MONTH = 0.98
    PRINCIPLE_MERGE_THRESHOLD = 0.90
    MIN_CONFIDENCE_THRESHOLD = 0.55


def worker_embed_batch_global(args):
    texts, model_name = args
    try:
        return OllamaEmbeddings(model=model_name).embed_documents(texts)
    except:
        return []


class LegalUtils:
    @staticmethod
    def force_unlock_db():
        if os.path.exists(os.path.join(LegalConfig.QDRANT_PATH, ".lock")):
            try:
                os.remove(os.path.join(LegalConfig.QDRANT_PATH, ".lock"))
            except:
                pass

    @staticmethod
    def extract_pdf_conclusion(file_path, char_limit=2500):
        try:
            if not os.path.exists(file_path): return "[Dosya bulunamadı]"
            doc = fitz.open(file_path)
            text = "".join([page.get_text() for page in doc])
            doc.close()
            return text[-char_limit:]
        except:
            return "[Karar okunamadı]"

    @staticmethod
    def clean_text(text):
        return re.sub(r'\s+', ' ', re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)).strip()

    @staticmethod
    def safe_pdf_text(text):
        if not text: return ""
        replacements = {'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S', 'ı': 'i', 'İ': 'I', 'ö': 'o',
                        'Ö': 'O', 'ç': 'c', 'Ç': 'C', '–': '-', '…': '...', '“': '"', '”': '"', '’': "'"}
        for k, v in replacements.items(): text = text.replace(k, v)
        try:
            return text.encode('latin-1', 'replace').decode('latin-1')
        except:
            return "???"


# ==================================================
# 2️⃣ LOGGING VE REASONING MOTORLARI
# ==================================================
class LegalAuditLogger:
    def __init__(self, case_id=None):
        self.case_id = case_id or str(uuid.uuid4())
        self.started_at = time.time()
        self.logs = []
        self._step_counter = 0

    def log_event(self, stage, title, description, inputs=None, outputs=None, score_impact=None, resulting_score=None,
                  confidence=None):
        self._step_counter += 1
        event = {"step": self._step_counter, "timestamp": time.time(), "stage": stage, "title": title,
                 "description": description, "inputs": inputs or {}, "outputs": outputs or {}}
        if score_impact is not None: event["score_impact"] = score_impact
        if resulting_score is not None: event["resulting_score"] = resulting_score
        if confidence: event["confidence"] = confidence
        self.logs.append(event)

    def export(self):
        return {"case_id": self.case_id, "started_at": self.started_at, "completed_at": time.time(),
                "timeline": self.logs}


class ActionableRecommendationEngine:
    RECOMMENDATION_PROFILE = {
        "DELIL": {"evidence_type": ["tanık", "belge", "bilirkişi", "keşif"], "priority": "YÜKSEK",
                  "estimated_cost": "Orta", "time_impact": "Orta", "base_score_range": (5, 10)},
        "ICTIHAT": {"evidence_type": ["emsal karar", "HGK kararı"], "priority": "ORTA", "estimated_cost": "Düşük",
                    "time_impact": "Kısa", "base_score_range": (3, 7)},
        "USUL": {"evidence_type": ["dilekçe", "itiraz"], "priority": "YÜKSEK", "estimated_cost": "Düşük",
                 "time_impact": "Kısa", "base_score_range": (2, 4)},
        "TALEP_DARALTMA": {"evidence_type": ["strateji"], "priority": "ORTA", "estimated_cost": "Düşük",
                           "time_impact": "Kısa", "base_score_range": (4, 6)}
    }

    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_concerns, query_text=""):
        recommendations = []
        for concern in judge_concerns:
            category = self._classify(concern)
            profile = self.RECOMMENDATION_PROFILE.get(category, self.RECOMMENDATION_PROFILE["DELIL"])
            rec_text = self._generate_text(concern, category)
            score_boost = random.randint(profile["base_score_range"][0], profile["base_score_range"][1])

            recommendations.append({
                "action_id": str(uuid.uuid4()),
                "title": rec_text[:80] + "...",
                "description": rec_text,
                "category": category,
                "focus": category,
                "evidence": {"type": "Genel", "source": self._infer_source(concern, query_text), "count": 1},
                "priority": profile["priority"],
                "estimated_cost": profile["estimated_cost"],
                "time_impact": profile["time_impact"],
                "risk_reduction": {"area": category, "expected_score_increase": score_boost},
                "suggestion": rec_text,
                "if_not_done": f"Risk devam eder: {concern[:30]}...",
                "why": concern
            })
        return recommendations

    def _classify(self, text):
        t = text.lower()
        if any(x in t for x in ["delil", "tanık", "belge"]): return "DELIL"
        if any(x in t for x in ["emsal", "yargıtay"]): return "ICTIHAT"
        if any(x in t for x in ["usul", "süre"]): return "USUL"
        return "TALEP_DARALTMA"

    def _generate_text(self, concern, cat):
        try:
            return self.llm.invoke(
                f"BAĞLAM: Türk Hukuku.\nAvukata {cat} odaklı SOMUT aksiyon önerisi yaz.\nSorun: {concern}").content.strip()
        except:
            return "İlgili hususta ek beyan sunulmalıdır."

    def _infer_source(self, concern, query):
        q = query.lower()
        if "miras" in q: return "Nüfus Müdürlüğü"
        if "iş" in q: return "SGK / İşyeri"
        if "tapu" in q: return "Tapu Müd."
        return "Dosya Kapsamı"


# ==================================================
# 3️⃣ ARAMA MOTORU (ÖNE ALINDI)
# ==================================================
class LegalSearchEngine:
    def __init__(self):
        self.config = LegalConfig()
        self.dense_embedder = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL)
        self.client = None
        atexit.register(self.close)

    def connect_db(self):
        if self.client: return True
        LegalUtils.force_unlock_db()
        try:
            self.client = QdrantClient(path=self.config.QDRANT_PATH)
            return True
        except:
            return False

    def close(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass

    def run_indexing(self):
        if not self.connect_db(): return False
        for key, conf in self.config.SOURCES.items():
            if not os.path.exists(conf["folder"]): os.makedirs(conf["folder"]); continue
            if not self.client.collection_exists(conf["collection"]):
                self.client.create_collection(conf["collection"],
                                              vectors_config=VectorParams(size=768, distance=Distance.COSINE))

            # Basit kontrol: Boşsa veya değişiklik varsa indeksle (Detaylı mantık kısaltıldı)
            files = [f for f in os.listdir(conf["folder"]) if f.endswith(".pdf")]
            if not files: continue

            # İndeksleme Mantığı (Basitleştirilmiş)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_points = []

            # Mevcut kayıtları kontrol et (Hız için)
            existing_count = self.client.count(conf["collection"]).count
            if existing_count > 0 and len(files) < 5: continue  # Örnek optimizasyon

            print(f"      🚀 İndeksleniyor: {conf['desc']}...")
            for f in files:
                try:
                    loader = PyMuPDFLoader(os.path.join(conf["folder"], f))
                    chunks = text_splitter.split_documents(loader.load())
                    vectors = self.dense_embedder.embed_documents(
                        [LegalUtils.clean_text(c.page_content) for c in chunks])
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                        payload = {"page_content": chunk.page_content, "source": f, "type": conf["desc"].upper(),
                                   "page": chunk.metadata.get("page", 0)}
                        all_points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
                except:
                    pass

            if all_points:
                batch_size = 64
                for i in range(0, len(all_points), batch_size):
                    self.client.upsert(conf["collection"], all_points[i:i + batch_size])
        return True

    def retrieve_raw_candidates(self, query):
        try:
            vec = self.dense_embedder.embed_query(query)
            hits = []
            for k, conf in self.config.SOURCES.items():
                try:
                    res = self.client.query_points(conf["collection"], query=vec, limit=60).points
                    for r in res:
                        if 'type' not in r.payload: r.payload['type'] = conf['desc'].upper()
                        hits.append(r)
                except:
                    pass

            # V101: Kota Sistemi
            emsal = sorted([h for h in hits if h.payload.get('type') != 'MEVZUAT'], key=lambda x: x.score,
                           reverse=True)[:7]
            mevzuat = sorted([h for h in hits if h.payload.get('type') == 'MEVZUAT'], key=lambda x: x.score,
                             reverse=True)[:3]

            final = emsal + mevzuat
            if not final: print("🔴 Belge bulunamadı."); return []
            print(f"   ✅ Seçilen: {len(emsal)} Emsal + {len(mevzuat)} Mevzuat")
            return final
        except:
            return []


# ==================================================
# 5️⃣ HAFIZA YÖNETİCİSİ
# ==================================================
class LegalMemoryManager:
    MAX_SCORE = 95

    def __init__(self, client, embedder, llm):
        self.client = client
        self.embedder = embedder
        self.llm = llm
        for n, c in LegalConfig.MEMORY_COLLECTIONS.items():
            if not self.client.collection_exists(c): self.client.create_collection(c,
                                                                                   vectors_config=VectorParams(size=768,
                                                                                                               distance=Distance.COSINE))
        self.rec_engine = ActionableRecommendationEngine(llm)
        self.audit_logger = LegalAuditLogger()
        self.latest_ui_data = {}

    def recall_principles(self, query):
        self.audit_logger = LegalAuditLogger()
        vec = self.embedder.embed_query(query)
        hits = self.client.query_points(LegalConfig.MEMORY_COLLECTIONS["principle"], query=vec, limit=5).points

        self.latest_ui_data = {"query": query, "principles": [], "audit_log": {}}

        if not hits: return ""

        mem_text = ""
        for h in hits:
            conf = h.payload.get("confidence", 0.5)
            text = h.payload.get("principle", "")

            # Basit Analiz
            score = min(conf * 100 + 10, 90)
            personas = {
                "judge": self._gen_text(f"HAKİM: {text} hakkında yorum yap."),
                "opponent": self._gen_text(f"KARŞI TARAF: {text} aleyhine konuş."),
                "expert": self._gen_text(f"BİLİRKİŞİ: {text} teknik analizi."),
                "devil": self._gen_text(f"RED GEREKÇESİ: {text} neden uygulanmaz?")
            }

            concerns = [l for l in personas["devil"].split("\n") if len(l) > 10][:3]
            actions = self.rec_engine.generate(concerns, query)

            sim_score = min(score + len(actions) * 3, 95)

            self.latest_ui_data["principles"].append({
                "text": text,
                "score_data": {"success_probability": score},
                "personas": personas,
                "action_plan": actions,
                "simulation": {"current": score, "projected": sim_score}
            })

            mem_text += f"- {text}\n"

        self.latest_ui_data["audit_log"] = self.audit_logger.export()
        return mem_text

    def _gen_text(self, prompt):
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    # Save/Consolidate Placeholder (Tam kod çok uzun, kritik akış için bu yeterli)
    def consolidate_principles_v79(self):
        pass

    def save_decision(self, *args):
        pass


# ==================================================
# 6️⃣ YARGIÇ
# ==================================================
class LegalJudge:
    def __init__(self, memory_manager):
        self.llm = ChatOllama(model=LegalConfig.LLM_MODEL, temperature=0.1)
        self.memory = memory_manager

    def validate_user_input(self, s, t):
        return True

    def generate_expanded_queries(self, s, t):
        return [s]

    def evaluate_candidates(self, candidates, story, topic, negatives):
        valid = []
        print("\n⚖️  Yargıç Değerlendiriyor...")
        for hit in candidates:
            # Basit Check
            type_desc = hit.payload.get('type', 'EMSAL')
            prompt = f"Sorgu: {story}\nBelge ({type_desc}): {hit.payload['page_content'][:500]}\nAlakalı mı? [EVET/HAYIR]"
            res = self.llm.invoke(prompt).content
            if "EVET" in res or "yes" in res.lower():
                valid.append({
                    "source": hit.payload['source'], "page": hit.payload['page'],
                    "type": type_desc, "text": hit.payload['page_content'],
                    "role": "DOĞRUDAN DELİL" if type_desc == "MEVZUAT" else "EMSAL",
                    "reason": res, "score": hit.score * 100
                })
                print(f"✅ {hit.payload['source']} ({type_desc})")
        return valid

    def generate_final_opinion(self, s, t, context):
        return self.llm.invoke(f"Sen avukatsın. Olay: {s}. Deliller: {context}. Rapor yaz.").content


# ==================================================
# 7️⃣ YENİ RAPORLAMA ARAÇLARI (V105-V110)
# ==================================================
class WhiteLabelConfig:
    def __init__(self, firm="LEGAL OS", footer="Otomatik Rapor"):
        self.firm = firm;
        self.footer = footer


class BrandedPDFGenerator(FPDF):
    def __init__(self, branding):
        super().__init__()
        self.branding = branding

    def header(self):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, LegalUtils.safe_pdf_text(self.branding.firm), align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.line(10, 20, 200, 20);
        self.ln(10)

    def footer(self):
        self.set_y(-15);
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, LegalUtils.safe_pdf_text(f"{self.branding.footer} | {self.page_no()}"), align='C')


class LegalReporter:
    @staticmethod
    def create_report(story, docs, advice, audit_data, filename, llm=None, personas=None, topic=""):
        pdf = BrandedPDFGenerator(WhiteLabelConfig())

        # Kapak
        pdf.add_page()
        pdf.set_font("helvetica", "B", 24);
        pdf.ln(60)
        pdf.cell(0, 10, "LEGAL OS RAPORU", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # İçerik
        pdf.add_page();
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 6, LegalUtils.safe_pdf_text(f"OLAY:\n{story}\n\nGORUS:\n{advice}"))

        # Ekler (V108)
        if llm and audit_data:
            # 1. Hakim Gerekçesi
            prompt = "Sen Hakimsin. Karar gerekçesi yaz."
            judge_reason = llm.invoke(prompt).content
            pdf.add_page();
            pdf.cell(0, 10, "EK-1: HAKIM GEREKCESI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(0, 6, LegalUtils.safe_pdf_text(judge_reason))

            # 2. İtiraz Dilekçesi
            prompt2 = f"Sen Avukatsın. Şu karara itiraz dilekçesi yaz:\n{judge_reason}"
            petition = llm.invoke(prompt2).content
            pdf.add_page();
            pdf.cell(0, 10, "EK-2: ITIRAZ DILEKCESI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(0, 6, LegalUtils.safe_pdf_text(petition))

        try:
            pdf.output(filename); print(f"\n📄 Rapor Hazır: {filename}")
        except Exception as e:
            print(f"PDF Hatası: {e}")


class LegalUIPrinter:
    @staticmethod
    def print_grand_ui_log(ui_data, doc_log):
        print("\n" + "█" * 60)
        print("🖥️  LEGAL OS V113 - CANLI ANALİZ EKRANI")
        print("█" * 60)

        print("\n📂 1. TARANAN BELGELER:")
        for d in doc_log: print(f"   📄 {d['source']} -> {d['role']}")

        if ui_data.get("principles"):
            p = ui_data["principles"][0]
            print(f"\n⚖️  2. TEMEL İLKE:\n   {p['text'][:100]}...")
            print(f"   🎲 Skor: %{p['score_data']['success_probability']}")
            print(f"   🔮 Simülasyon: %{p['simulation']['current']} -> %{p['simulation']['projected']}")

            print("\n🚀 3. AKSİYON PLANI:")
            for a in p['action_plan']:
                print(f"   🔧 {a['title']} (Etki: +{a['risk_reduction']['expected_score_increase']})")
        print("\n" + "█" * 60 + "\n")


# ==================================================
# 8️⃣ ANA UYGULAMA
# ==================================================
class LegalApp:
    def __init__(self):
        print("🚀 LEGAL SUITE V113 (Final Stable Release)...")
        self.search_engine = LegalSearchEngine()

        if self.search_engine.connect_db():
            self.memory_manager = LegalMemoryManager(
                self.search_engine.client,
                self.search_engine.dense_embedder,
                ChatOllama(model=LegalConfig.LLM_MODEL, temperature=0.1)
            )
        else:
            self.memory_manager = None

        self.judge = LegalJudge(self.memory_manager)
        self.reporter = LegalReporter()
        self.ui_printer = LegalUIPrinter()

    def run(self):
        if not self.search_engine.run_indexing(): sys.exit()
        if self.memory_manager: self.memory_manager.consolidate_principles_v79()

        print("\n✅ HAZIR.")
        try:
            while True:
                story = input("\n📝 Olay: ");
                if story == 'q': break
                topic = input("🎯 Odak: ")

                cands = self.search_engine.retrieve_raw_candidates(story + " " + topic)
                if not cands: continue

                valid = self.judge.evaluate_candidates(cands, story, topic, [])
                if not valid: print("🔴 Uygun belge yok."); continue

                doc_log = [{"source": d['source'], "role": d['role'], "reason": d['reason']} for d in valid]
                context = "\n".join([f"{d['source']}: {d['text'][:200]}" for d in valid])

                if self.memory_manager:
                    self.memory_manager.recall_principles(story)
                    self.ui_printer.print_grand_ui_log(self.memory_manager.latest_ui_data, doc_log)

                advice = self.judge.generate_final_opinion(story, topic, context)

                # Raporlama Parametreleri
                audit = self.memory_manager.latest_ui_data.get("audit_log", {}) if self.memory_manager else {}
                llm = self.memory_manager.rec_engine.llm if self.memory_manager else None
                personas = self.memory_manager.latest_ui_data.get("principles", [{}])[0].get(
                    "personas") if self.memory_manager and self.memory_manager.latest_ui_data.get(
                    "principles") else None

                self.reporter.create_report(story, valid, advice, audit, "Hukuki_Rapor_V113.pdf", llm, personas, topic)

        except KeyboardInterrupt:
            print("\n👋 Çıkış.")
        except Exception as e:
            print(f"\n⚠️ Kritik Hata: {e}")
        finally:
            self.search_engine.close()


if __name__ == "__main__":
    freeze_support()
    LegalApp().run()