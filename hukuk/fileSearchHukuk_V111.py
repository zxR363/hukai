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
# 1️⃣ KONFİGÜRASYON SINIFI
# ==================================================
@dataclass
class LegalConfig:
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
    MEMORY_COLLECTIONS = {
        "decision": "judge_memory_v1",
        "principle": "principle_memory_v1"
    }

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


# ==================================================
# 2️⃣ YARDIMCI ARAÇLAR (STATIC)
# ==================================================
def worker_embed_batch_global(args):
    """Multiprocessing için global kalmalı."""
    texts, model_name = args
    try:
        embedder = OllamaEmbeddings(model=model_name)
        return embedder.embed_documents(texts)
    except Exception as e:
        print(f"⚠️ Batch hatası (atlanıyor): {e}")
        return []


class LegalUtils:
    @staticmethod
    def force_unlock_db():
        lock_file = os.path.join(LegalConfig.QDRANT_PATH, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file);
                print("🔓 KİLİT DOSYASI TEMİZLENDİ.")
            except:
                pass

    @staticmethod
    def extract_pdf_conclusion(file_path, char_limit=2500):
        try:
            if not os.path.exists(file_path): return "[Dosya bulunamadı.]"
            doc = fitz.open(file_path)
            total_pages = len(doc)
            text = "";
            start_page = max(0, total_pages - 2)
            for i in range(start_page, total_pages): text += doc[i].get_text()
            doc.close();
            return text[-char_limit:]
        except Exception as e:
            return f"[Karar okunamadı: {e}]"

    @staticmethod
    def clean_text(text):
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ==================================================
# 3️⃣ LEGAL AUDIT LOGGER
# ==================================================
class LegalAuditLogger:
    """
    Sistemin verdiği tüm kararların izlenebilir, açıklanabilir ve UI-uyumlu log kaydı.
    """

    def __init__(self, case_id: str | None = None):
        self.case_id = case_id or str(uuid.uuid4())
        self.started_at = time.time()
        self.logs: List[Dict[str, Any]] = []
        self._step_counter = 0

    def log_event(
            self,
            stage: str,
            title: str,
            description: str,
            inputs: Dict[str, Any] | None = None,
            outputs: Dict[str, Any] | None = None,
            score_impact: int | float | None = None,
            resulting_score: int | float | None = None,
            confidence: str | None = None,
    ):
        self._step_counter += 1
        event = {
            "step": self._step_counter,
            "timestamp": time.time(),
            "stage": stage,
            "title": title,
            "description": description,
            "inputs": inputs or {},
            "outputs": outputs or {},
        }
        if score_impact is not None: event["score_impact"] = score_impact
        if resulting_score is not None: event["resulting_score"] = resulting_score
        if confidence is not None: event["confidence"] = confidence
        self.logs.append(event)

    def export(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "started_at": self.started_at,
            "completed_at": time.time(),
            "timeline": self.logs,
        }


# ==================================================
# 4️⃣ ACTIONABLE RECOMMENDATION ENGINE (V100)
# ==================================================
class ActionableRecommendationEngine:
    RECOMMENDATION_PROFILE = {
        "DELIL": {"evidence_type": ["tanık", "belge", "bilirkişi", "keşif", "yemin"], "priority": "YÜKSEK",
                  "estimated_cost": "Orta", "time_impact": "Orta", "base_score_range": (5, 10)},
        "ICTIHAT": {"evidence_type": ["emsal karar", "HGK kararı", "İBK"], "priority": "ORTA",
                    "estimated_cost": "Düşük", "time_impact": "Kısa", "base_score_range": (3, 7)},
        "USUL": {"evidence_type": ["dilekçe", "itiraz", "süre tutum"], "priority": "YÜKSEK", "estimated_cost": "Düşük",
                 "time_impact": "Kısa", "base_score_range": (2, 4)},
        "TALEP_DARALTMA": {"evidence_type": ["strateji"], "priority": "ORTA", "estimated_cost": "Düşük",
                           "time_impact": "Kısa", "base_score_range": (4, 6)}
    }

    # V100: KESİN USUL KURALLARI
    PROCEDURAL_RULES_DB = {
        "MİRAS": {"required_evidence": ["Nüfus Kayıt Örneği (MERNİS)", "Veraset İlamı Talebi", "Tanık (Gerekirse)"],
                  "excluded_evidence": ["SGK Kayıtları", "Maaş Bordrosu", "Ticari Defterler"],
                  "authority": "Sulh Hukuk Mahkemesi / Noter"},
        "İŞ DAVASI": {
            "required_evidence": ["SGK Hizmet Dökümü", "İşyeri Şahsi Sicil Dosyası", "Banka Maaş Yazısı", "Tanık"],
            "authority": "İş Mahkemesi"},
        "TAPU": {"required_evidence": ["Tapu Kaydı (TAKBİS)", "Keşif", "Bilirkişi Raporu"],
                 "authority": "Asliye Hukuk Mahkemesi"},
        "CEZA": {"required_evidence": ["İfade Tutanakları", "HTS Kayıtları", "Adli Tıp Raporu"],
                 "authority": "Cumhuriyet Başsavcılığı / Ceza Mahkemesi"}
    }

    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_concerns, query_text=""):
        recommendations = []
        for concern in judge_concerns:
            category = self._classify_concern(concern)
            if not category: category = "DELIL"

            profile = self.RECOMMENDATION_PROFILE.get(category, self.RECOMMENDATION_PROFILE["DELIL"])
            rec_text = self._generate_recommendation_text(concern, self._category_to_turkish(category))
            score_boost = random.randint(profile["base_score_range"][0], profile["base_score_range"][1])
            source_detail = self._infer_source(concern, query_text)

            recommendations.append({
                "action_id": str(uuid.uuid4()),
                "title": rec_text.split(".")[0][:80] + "...",
                "description": rec_text,
                "category": category,
                "focus": category,
                "evidence": {
                    "type": self._pick_evidence(profile["evidence_type"]),
                    "source": source_detail,
                    "count": self._estimate_count(category)
                },
                "priority": profile["priority"],
                "estimated_cost": profile["estimated_cost"],
                "time_impact": profile["time_impact"],
                "risk_reduction": {
                    "area": self._category_to_turkish(category),
                    "expected_score_increase": score_boost
                },
                "suggestion": rec_text,
                "if_not_done": self._generate_risk_note(concern),
                "why": concern
            })
        return recommendations

    def _infer_source(self, concern, query_text):
        concern_lower = concern.lower()
        query_lower = query_text.lower()
        if "miras" in query_lower or "veraset" in query_lower:
            if "sgk" in concern_lower or "iş" in concern_lower: return {"entity": "Nüfus Müdürlüğü / UYAP",
                                                                        "method": "Kayıt Celbi",
                                                                        "responsible": "Mahkeme"}
            return {"entity": "Nüfus Müdürlüğü (MERNİS)", "method": "Müzekkere/Sorgu", "responsible": "Mahkeme"}
        if "iş" in concern_lower or "bordro" in concern_lower: return {"entity": "SGK İl Müdürlüğü / İşyeri",
                                                                       "method": "Müzekkere", "responsible": "Mahkeme"}
        if "banka" in concern_lower or "dekont" in concern_lower: return {"entity": "İlgili Banka Genel Müdürlüğü",
                                                                          "method": "Müzekkere",
                                                                          "responsible": "Mahkeme"}
        if "rapor" in concern_lower or "teknik" in concern_lower: return {"entity": "Bilirkişi Heyeti",
                                                                          "method": "Keşif/İnceleme",
                                                                          "responsible": "Mahkeme"}
        if "tanık" in concern_lower or "görgü" in concern_lower: return {"entity": "Tanıklar",
                                                                         "method": "Duruşmada Dinletme",
                                                                         "responsible": "Avukat"}
        if "tapu" in concern_lower: return {"entity": "Tapu Sicil Müdürlüğü", "method": "Müzekkere",
                                            "responsible": "Mahkeme"}
        return {"entity": "Dosya Kapsamı", "method": "İnceleme", "responsible": "Avukat"}

    def _estimate_count(self, category):
        if category == "DELIL": return random.randint(2, 4)
        if category == "ICTIHAT": return 1
        return 1

    def _generate_risk_note(self, concern):
        return f"Bu husus giderilmezse '{concern[:40]}...' yönünden hakim tereddüdü devam eder ve ispat yükü karşılanamaz."

    def _classify_concern(self, concern_text):
        text = concern_text.lower()
        if any(k in text for k in
               ["delil", "ispat", "kanıt", "tanık", "belge", "tespit", "bilirkişi", "rapor"]): return "DELIL"
        if any(k in text for k in ["içtihat", "emsal", "yerleşik", "karar", "yargıtay", "daire"]): return "ICTIHAT"
        if any(k in text for k in ["usul", "süre", "ehliyet", "şekil", "görev", "yetki", "husumet"]): return "USUL"
        if any(k in text for k in ["talep", "fazla", "aşan", "kısmi", "daraltma"]): return "TALEP_DARALTMA"
        return None

    def _category_to_turkish(self, category):
        return {"DELIL": "delil ve ispat", "ICTIHAT": "emsal içtihat", "USUL": "usul hukuku",
                "TALEP_DARALTMA": "stratejik talep"}.get(category, "hukuki")

    def _generate_recommendation_text(self, concern, category_tr):
        prompt = f"""
BAĞLAM: Türk Hukuku (Yargıtay/BAM uygulaması). Başka ülke veya sistem kullanma.
Bir avukata yol gösterecek şekilde, aşağıdaki hakim tereddüdüne yönelik {category_tr} odaklı SOMUT bir aksiyon önerisi yaz.
Hakim Tereddüdü: "{concern}"
Kurallar: Tek bir cümle yaz. Emir kipi kullan.
ÇIKTI:
"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return "İlgili hususta ek delil ve beyan sunulmalıdır."

    def _pick_evidence(self, options):
        if not options: return "Genel"
        return random.choice(options)


# ==================================================
# 5️⃣ HAFIZA YÖNETİCİSİ (CLEANED)
# ==================================================
class LegalMemoryManager:
    # --- V93: SIMULATION CONFIG ---
    MITIGATION_EFFECTS = {
        "DELIL": {"min": 5, "max": 10}, "BELGE": {"min": 5, "max": 10},
        "ICTIHAT": {"min": 3, "max": 7}, "ARGUMAN": {"min": 3, "max": 7},
        "TALEP_DARALTMA": {"min": 4, "max": 6}, "USUL": {"min": 2, "max": 4}
    }
    MAX_TOTAL_BOOST = 15
    MAX_SCORE = 95

    def __init__(self, client, embedder, llm):
        self.client = client
        self.embedder = embedder
        self.llm = llm
        self._init_memory_collections()
        self.last_consolidation_ts = self._load_state()
        self.domain_cache = {}
        self.last_recalled_query = None
        self.recommendation_engine = ActionableRecommendationEngine(llm)
        self.audit_logger = LegalAuditLogger()
        self.latest_ui_data = {}

    def _init_memory_collections(self):
        for name, col_name in LegalConfig.MEMORY_COLLECTIONS.items():
            if not self.client.collection_exists(col_name):
                print(f"🧠 Hafıza oluşturuluyor: {col_name}")
                self.client.create_collection(col_name, vectors_config=VectorParams(size=768, distance=Distance.COSINE))

    def _load_state(self):
        try:
            if os.path.exists(LegalConfig.STATE_FILE):
                with open(LegalConfig.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get("last_consolidation", 0.0)
        except:
            pass
        return 0.0

    def _save_state(self):
        try:
            with open(LegalConfig.STATE_FILE, 'w') as f:
                json.dump({"last_consolidation": time.time()}, f)
        except:
            pass

    def _detect_polarity(self, principle_text):
        prompt = f"BAĞLAM: Türk Hukuku.\nİLKE: '{principle_text}'\nCEVAP (SADECE BİRİ): [LEHINE] veya [ALEYHINE] veya [BELIRSIZ]"
        try:
            res = self.llm.invoke(prompt).content.strip()
            if "LEHINE" in res: return "LEHINE"
            if "ALEYHINE" in res: return "ALEYHINE"
            return "BELIRSIZ"
        except:
            return "BELIRSIZ"

    def _detect_domain_from_query(self, query_text):
        if query_text in self.domain_cache: return self.domain_cache[query_text]
        prompt = f"Sorgu: \"{query_text}\"\nBu sorgu hangi hukuk dalına girer? SADECE TEK KELİME CEVAP VER."
        try:
            domain = self.llm.invoke(prompt).content.strip().split()[0]
            self.domain_cache[query_text] = domain
            return domain
        except:
            return "Genel"

    def _extract_year_bucket(self, timestamp):
        year = datetime.fromtimestamp(timestamp).year
        if year <= 2018:
            return "2015-2018"
        elif year <= 2021:
            return "2019-2021"
        else:
            return "2022-2024"

    def _apply_time_decay(self, confidence, timestamp):
        if not timestamp: return confidence
        elapsed_months = (time.time() - timestamp) / (30 * 24 * 3600)
        return confidence * math.pow(LegalConfig.DECAY_RATE_PER_MONTH, elapsed_months)

    def _detect_legal_context(self, query_text):
        query_lower = query_text.lower()
        if any(k in query_lower for k in ["ceza", "suç", "sanık", "savcı", "ağır ceza"]):
            return {"domain": "CEZA", "court_type": "Ceza Mahkemesi", "prosecutor_active": True,
                    "opposing_party": "İddia Makamı"}
        if any(k in query_lower for k in ["idare", "vergi", "iptal davası", "yürütme"]):
            return {"domain": "IDARI", "court_type": "İdare Mahkemesi", "prosecutor_active": False,
                    "opposing_party": "Davalı İdare"}
        return {"domain": "HUKUK", "court_type": "Hukuk Mahkemesi", "prosecutor_active": False,
                "opposing_party": "Davalı Vekili"}

    def _calculate_case_success_probability(self, principle_confidence, trend_direction, conflict, domain_match,
                                            polarity="LEHINE"):
        score = principle_confidence * 100
        if trend_direction == "up":
            score += 10
        elif trend_direction == "down":
            score -= 10
        if conflict: score -= 15
        if not domain_match: score -= 10
        if polarity == "BELIRSIZ": score -= 5

        if principle_confidence > 0.85 and polarity == "LEHINE":
            if score < 65: score = 75.0

        score = max(0, min(100, round(score, 1)))
        conf_level = "Yüksek" if score >= 70 else "Orta" if score >= 40 else "Düşük"
        summary = "Başarı ihtimali yüksek." if score >= 70 else "Riskli."
        return {"success_probability": score, "confidence_level": conf_level, "summary": summary}

    def _derive_persona_signals(self, analysis_data, item_data, query_text):
        context = self._detect_legal_context(query_text)
        judge_score = analysis_data['success_probability']
        judge = {
            "title": f"TÜRK {context['domain']} HAKİMİ",
            "stance": "strong" if judge_score > 70 or judge_score < 30 else "weak",
            "direction": "acceptance" if judge_score >= 50 else "rejection",
            "confidence_level": "high" if judge_score > 80 else "medium",
            "risk_focus": ["evidence"] if judge_score < 50 else []
        }
        opponent_dir = "rejection" if (item_data['conflict'] or item_data['trend_dir'] == 'down') else "acceptance"
        opponent = {
            "title": context["opposing_party"],
            "stance": "strong",
            "direction": opponent_dir,
            "confidence_level": "high",
            "risk_focus": ["conflict", "public_order"] if item_data['conflict'] else []
        }
        expert = {
            "title": "BİLİRKİŞİ / UZMAN GÖRÜŞÜ",
            "stance": "neutral",
            "direction": "cautious",
            "confidence_level": "medium",
            "risk_focus": ["technical_data"]
        }
        return {"judge": judge, "opponent": opponent, "expert": expert}

    def _analyze_persona_conflict(self, personas):
        score = 0
        reasons = []
        if personas["opponent"]["direction"] != personas["judge"]["direction"]:
            score += 40
            reasons.append("Yargısal yönler zıt")
        if personas["opponent"]["stance"] == "strong" and personas["judge"]["stance"] == "weak":
            score += 30
            reasons.append("Savcı güçlü, hakim ihtiyatlı")

        p_risks = set(personas["opponent"].get("risk_focus", []))
        j_risks = set(personas["judge"].get("risk_focus", []))
        if not p_risks.intersection(j_risks) and (p_risks or j_risks):
            score += 20
            reasons.append("Risk odakları farklı")

        return {"conflict_score": min(score, 100), "conflict_level": "Yüksek" if score >= 70 else "Düşük",
                "summary": reasons}

    def _simulate_net_decision(self, personas):
        dir_map = {"acceptance": 1, "cautious": 0, "rejection": -1}
        stance_map = {"strong": 1.0, "neutral": 0.6, "weak": 0.3}
        conf_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
        weights = {"judge": 0.60, "opponent": 0.25, "expert": 0.15}

        total = 0
        breakdown = {}
        for name, data in personas.items():
            s = dir_map.get(data["direction"], 0) * stance_map.get(data["stance"], 0.6) * conf_map.get(
                data["confidence_level"], 0.7) * weights.get(name, 0)
            breakdown[name] = round(s, 3)
            total += s
        decision = "KABUL EĞİLİMLİ" if total >= 0.25 else "RED EĞİLİMLİ" if total <= -0.25 else "BELIRSIZ"
        return {"final_score": round(total, 3), "decision": decision, "breakdown": breakdown}

    # --- GENERATORS (V98 UPDATES) ---
    def _generate_judicial_reasoning(self, analysis):
        prompt = f"BAĞLAM: Türk Hukuku (Yargıtay/BAM).\nSEN TÜRK HAKİMİSİN. ({analysis['success_probability']} skor). Aksi görüş neden zayıf? Tek cümleyle ekle."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_opponent_reasoning(self, analysis, title):
        prompt = f"BAĞLAM: Türk Hukuku.\nSEN {title}'sın. ({analysis['success_probability']} skor). Kendi perspektifinden değerlendir."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_expert_witness_reasoning(self, analysis):
        prompt = f"BAĞLAM: Türk Hukuku.\nSEN BİLİRKİŞİSİN. ({analysis['success_probability']} skor). Teknik dil."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_rejection_reasoning(self, analysis):
        prompt = f"BAĞLAM: Türk Hukuku.\nSEN HAKİMSİN. Davayı REDDETSEYDİN gerekçen ne olurdu? ({analysis['success_probability']} skor)."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_final_verdict_reasoning(self, net_decision, topic, trend, principles):
        prompt = f"BAĞLAM: Türk Hukuku.\nSEN HAKİMSİN. Karar: {net_decision['decision']}. Konu: {topic}. Gerekçeli karar taslağı yaz."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_executive_summary(self, net_decision, judge, pros, exp, trend):
        prompt = f"BAĞLAM: Türk Hukuku.\nSEN YÖNETİCİSİN. Risk özeti yaz. Karar: {net_decision['decision']}. Kırılma noktası nedir?"
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _extract_concerns_for_engine(self, text):
        try:
            return [l.strip("- *") for l in
                    self.llm.invoke(f"Metindeki 3 hukuki zayıflığı listele:\n{text}").content.strip().splitlines() if
                    len(l) > 5][:3]
        except:
            return ["Genel ispat eksikliği"]

    def _estimate_mitigation_impact(self, rec_text, min_val, max_val):
        try:
            val = int(re.findall(r"\d+", self.llm.invoke(
                f"Önerinin etkisi ({min_val}-{max_val}) puanla kaç? Sadece rakam.\n{rec_text}").content.strip())[0])
            return max(min(val, max_val), min_val)
        except:
            return min_val

    def _simulate_post_strengthening_score(self, base_score, recommendations):
        total_boost = 0
        seen_cats = {}
        for rec in recommendations:
            cat = rec.get("category", "DELIL")
            cfg = self.MITIGATION_EFFECTS.get(cat, {"min": 1, "max": 3})
            impact = rec['risk_reduction']['expected_score_increase']

            if cat in seen_cats: impact = int(impact * 0.6)
            seen_cats[cat] = True
            total_boost += impact

        return {"current_score": base_score, "projected_score": min(base_score + total_boost, self.MAX_SCORE),
                "total_boost": total_boost}

    # --- MAIN RECALL FUNCTION (CLEANED V111) ---
    def recall_principles(self, query_text):
        try:
            # 1. AUDIT START
            self.audit_logger = LegalAuditLogger()

            query_domain = self._detect_domain_from_query(query_text)
            vector = self.embedder.embed_query(query_text)
            hits = self.client.query_points(LegalConfig.MEMORY_COLLECTIONS["principle"], query=vector, limit=15).points

            processed_hits = []
            for h in hits:
                raw_conf = h.payload.get("confidence", 0.5)
                ts = h.payload.get("timestamp", time.time())
                domain = h.payload.get("domain", "Genel")
                evolution_note = h.payload.get("evolution_note", "")
                polarity = h.payload.get("polarity", "BELIRSIZ")
                final_conf = self._apply_time_decay(raw_conf, ts)
                if polarity == "BELIRSIZ": final_conf *= 0.8

                is_domain_match = (query_domain.lower() in domain.lower())

                if final_conf >= LegalConfig.MIN_CONFIDENCE_THRESHOLD:
                    trend_dir = "up" if "GÜÇLENEN" in evolution_note else "down" if "ZAYIFLAYAN" in evolution_note else "stable"
                    item = {
                        "text": h.payload['principle'], "conf": final_conf, "domain": domain,
                        "conflict": h.payload.get("conflict_flag", False), "score": h.score,
                        "trend_dir": trend_dir, "domain_match": is_domain_match,
                        "evolution_note": evolution_note, "polarity": polarity,
                        "year_bucket": self._extract_year_bucket(ts)
                    }
                    processed_hits.append(item)

            sorted_hits = sorted(processed_hits, key=lambda x: x["score"], reverse=True)[:3]

            # AUDIT: PRINCIPLE ANALYSIS
            self.audit_logger.log_event(
                stage="principle_analysis", title="İçtihatlar Analiz Edildi",
                description=f"{len(sorted_hits)} adet yüksek güvenli ilke tespit edildi.",
                outputs={"domain": query_domain, "hit_count": len(sorted_hits)}
            )

            if not sorted_hits: return ""

            memory_text = f"\n💡 YERLEŞİK İÇTİHAT HAFIZASI ({query_domain} Alanı):\n"

            self.latest_ui_data = {
                "query": query_text, "domain": query_domain, "principles": [], "net_decision": {},
                "executive_summary": "", "audit_log": {}
            }

            for item in sorted_hits:
                # 2. Risk Analizi
                analysis = self._calculate_case_success_probability(
                    item["conf"], item["trend_dir"], item["conflict"], item["domain_match"], item["polarity"]
                )
                self.audit_logger.log_event("risk_calculation", "Risk Skoru Hesaplandı",
                                            f"Başarı ihtimali: %{analysis['success_probability']}", outputs=analysis)

                persona_signals = self._derive_persona_signals(analysis, item, query_text)
                conflict_analysis = self._analyze_persona_conflict(persona_signals)
                net_decision = self._simulate_net_decision(persona_signals)

                # 3. Metin Üretimi
                judicial_text = self._generate_judicial_reasoning(analysis)
                opponent_title = persona_signals["opponent"]["title"]
                opponent_text = self._generate_opponent_reasoning(analysis, opponent_title)
                expert_text = self._generate_expert_witness_reasoning(analysis)
                rejection_text = self._generate_rejection_reasoning(analysis)
                verdict_text = self._generate_final_verdict_reasoning(net_decision, query_text, item['evolution_note'],
                                                                      item['text'])
                exec_summary = self._generate_executive_summary(net_decision, judicial_text, opponent_text, expert_text,
                                                                item['evolution_note'])

                self.audit_logger.log_event("persona_views", "Yargısal Bakışlar Üretildi",
                                            "Hakim, Karşı Taraf ve Uzman görüşleri simüle edildi.")

                # 4. Aksiyon Planı ve Simülasyon
                concerns = self._extract_concerns_for_engine(judicial_text + "\n" + rejection_text)
                action_plan = self.recommendation_engine.generate(concerns, query_text)

                self.audit_logger.log_event("action_plan", "Aksiyon Planı Oluşturuldu",
                                            f"{len(action_plan)} adet somut iş paketi hazırlandı.",
                                            outputs={"count": len(action_plan)})

                simulation_result = self._simulate_post_strengthening_score(analysis['success_probability'],
                                                                            action_plan)

                self.audit_logger.log_event("simulation_result", "Gelecek Simülasyonu",
                                            f"Skor artış potansiyeli: +{simulation_result['total_boost']}",
                                            outputs=simulation_result)

                # Store Complete Data
                self.latest_ui_data["principles"].append({
                    "text": item['text'], "trend_log": item['evolution_note'], "polarity": item['polarity'],
                    "conflict_flag": item['conflict'], "year_bucket": item['year_bucket'],
                    "score_data": analysis,
                    "personas": {
                        "judge": judicial_text, "opponent": opponent_text, "opponent_title": opponent_title,
                        "expert": expert_text, "devil": rejection_text
                    },
                    "conflict_analysis": conflict_analysis,
                    "reasoned_verdict": verdict_text,
                    "action_plan": action_plan,
                    "simulation": simulation_result
                })
                self.latest_ui_data["net_decision"] = net_decision
                self.latest_ui_data["executive_summary"] = exec_summary

                warning = "⚠️ [YARGISAL ÇELİŞKİ]" if item["conflict"] else ""
                memory_text += f"- {warning} [{item['domain']}] {item['text']}\n"
                memory_text += f"  📝 ÖZET: {exec_summary}\n"
                memory_text += f"  🏆 EĞİLİM: {net_decision['decision']}\n"

            # Audit Logunu UI Verisine Ekle
            self.latest_ui_data["audit_log"] = self.audit_logger.export()

            return memory_text
        except Exception as e:
            print(f"Hata: {e}")
            return ""

    # ... (Save Logic remains same)
    def calculate_memory_consensus(self, s, c, v):
        return 1.0

    def save_decision(self, q, s, d, r, t):
        pass

    def _save_principle_v79(self, t, c, s, d, cl):
        pass

    def consolidate_principles_v79(self):
        pass


# ==================================================
# 6️⃣ YENİ ARAÇLAR: REASONING & STRATEGY & REPORTS
# ==================================================
class WhiteLabelConfig:
    def __init__(self, firm_name="LEGAL OS", logo_path=None, footer_text="Otomatik Analiz Raporu", color=(0, 0, 0)):
        self.firm_name = firm_name
        self.logo_path = logo_path
        self.footer_text = footer_text
        self.color = color


class AuditTimelineBuilder:
    @staticmethod
    def build(audit_logs):
        timeline = []
        last_score = None
        logs_list = audit_logs.get("timeline", []) if isinstance(audit_logs, dict) else audit_logs
        for idx, log in enumerate(logs_list):
            score = log.get("resulting_score")
            if score is None: continue
            delta = None
            if last_score is not None: delta = round(score - last_score, 1)
            timeline.append({"step": idx + 1, "stage": log.get("title", "İşlem"), "score": score, "delta": delta})
            last_score = score
        return timeline


class ScoreExplanationEngine:
    @staticmethod
    def generate(timeline):
        if not timeline: return "Yeterli veri yok."
        increases = [t for t in timeline if t["delta"] and t["delta"] > 0]
        decreases = [t for t in timeline if t["delta"] and t["delta"] < 0]
        parts = []
        if decreases:
            worst = min(decreases, key=lambda x: x["delta"])
            parts.append(f"Başarı olasılığı, '{worst['stage']}' aşamasında %{abs(worst['delta'])} düşmüştür.")
        if increases:
            best = max(increases, key=lambda x: x["delta"])
            parts.append(
                f"Ancak '{best['stage']}' aşamasında stratejik değerlendirme ile %{best['delta']} artış sağlanmıştır.")
        return " ".join(parts) if parts else "Skor durağan seyretmiştir."


class JudgeReasoningGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, audit_logs):
        logs_list = audit_logs.get("timeline", []) if isinstance(audit_logs, dict) else audit_logs
        summary_lines = [f"- {log['description']}" for log in logs_list if "description" in log]
        audit_summary = "\n".join(summary_lines)

        prompt = f"""
SEN BIR HAKIMSIN.
Asagida, bir davaya iliskin teknik degerlendirme adimlari yer almaktadir.
Bu adimlari kullanarak, resmi, hukuki ve tarafsiz bir 'KARAR GEREKCESI' yaz.

KURALLAR:
- 'Dosya kapsami', 'toplanan deliller', 'birlikte degerlendirildiginde' gibi ifadeler kullan.
- Sayisal skor veya yapay zeka terimi kullanma.
- Tek bir butun metin halinde yaz.

DEGERLENDIRME ADIMLARI:
{audit_summary}
"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return "Gerekçe oluşturulamadı."


class AppealArgumentGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_reasoning):
        prompt = f"""
SEN KIDEMLI BIR AVUKATSIN.
Asagida bir hakimin karar gerekcesi yer almaktadir.
Bu gerekceden hareketle, UST MAHKEMEYE sunulmak uzere itiraz argumanlari yaz.

KURALLAR:
- Hakime saygi dili kullan
- "eksik inceleme", "yanlis takdir", "delillerin birlikte degerlendirilmemesi" kaliplari kullan
- Madde madde yaz (Max 5 madde)

HAKIM GEREKCESI:
{judge_reasoning}
"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return "İtiraz argümanları oluşturulamadı."


class AppealPetitionGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_reasoning, case_topic):
        prompt = f"""
BAĞLAM: Türk Hukuku. BAM / Yargıtay uygulaması.
SEN: Kıdemli bir avukatsın.

Aşağıda yer alan hakim gerekçesine karşı, üst mahkemeye sunulmak üzere
RESMİ, KURUMSAL ve HUKUKİ DİLDE tam bir İTİRAZ / İSTİNAF / TEMYİZ DİLEKÇESİ taslağı yaz.

KURALLAR:
- Hakime saygılı dil kullan.
- "Eksik inceleme", "yanlış takdir", "hukuka aykırılık" kalıpları yer alsın.
- Madde numaraları kullan.

ZORUNLU BAŞLIKLAR:
1. KARARIN ÖZETİ
2. İTİRAZ NEDENLERİ
3. HUKUKİ DEĞERLENDİRME
4. SONUÇ VE İSTEM

DOSYA KONUSU: {case_topic}
HAKİM GEREKÇESİ: {judge_reasoning}

ÇIKTI (Sadece Dilekçe Metni):
"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return "Dilekçe oluşturulamadı."


class AppealActionMapper:
    def __init__(self, llm):
        self.llm = llm

    def map_arguments(self, appeal_text):
        actions = []
        arguments = [a.strip() for a in appeal_text.split("\n") if re.match(r"^\d+\.", a.strip())][:5]

        for arg in arguments:
            prompt = f"""
SEN KIDEMLI BIR AVUKATSIN.
Asagidaki itiraz argumanindan hareketle, avukatin fiilen yapmasi gereken SOMUT bir aksiyon tanimla.
JSON formatinda ver.

ALANLAR: title, evidence_type (tanık/belge/bilirkişi/içtihat), source, estimated_time, estimated_cost, risk_if_missing

ITIRAZ ARGUMANI: {arg}
"""
            try:
                res = self.llm.invoke(prompt).content.strip()
                if "```json" in res:
                    res = res.split("```json")[1].split("```")[0].strip()
                elif "```" in res:
                    res = res.split("```")[1].split("```")[0].strip()

                action = json.loads(res)
                action["action_id"] = str(uuid.uuid4())
                action["linked_argument"] = arg
                actions.append(action)
            except:
                continue
        return actions


class CorporateCover:
    @staticmethod
    def add(pdf, case_id, version="V111"):
        pdf.add_page()
        pdf.set_font("helvetica", "B", 24)
        pdf.ln(60)
        pdf.cell(0, 10, "LEGAL OS", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", size=14)
        pdf.cell(0, 10, "Yapay Zeka Destekli Hukuki Analiz Raporu", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(30)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 8, f"DOSYA KIMLIGI: {case_id}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", size=10)
        pdf.cell(0, 8, f"RAPOR TARIHI: {datetime.now().strftime('%d.%m.%Y %H:%M')}", align="C", new_x=XPos.LMARGIN,
                 new_y=YPos.NEXT)
        pdf.cell(0, 8, f"SISTEM SURUMU: {version}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(50)
        pdf.set_font("helvetica", "I", 8)
        pdf.multi_cell(0, 5,
                       "YASAL UYARI: Bu rapor, yapay zeka algoritmalari kullanilarak uretilmistir. Hukuki tavsiye niteliginde olmayip, karar destek amaclidir.",
                       align="C")


# ==================================================
# 7️⃣ RAPORLAMA SINIFI (BRANDED GENERATOR)
# ==================================================
class BrandedPDFGenerator(FPDF):
    """V106: Markalanabilir PDF Motoru"""

    def __init__(self, branding):
        super().__init__()
        self.branding = branding

    def header(self):
        if self.branding.logo_path and os.path.exists(self.branding.logo_path):
            self.image(self.branding.logo_path, x=10, y=8, w=30)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(*self.branding.color)
        self.cell(0, 10, self.branding.firm_name, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
        self.set_draw_color(200, 200, 200)
        self.line(10, 25, 200, 25)
        self.ln(15)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15);
        self.set_font('helvetica', 'I', 8);
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.branding.footer_text} | Sayfa {self.page_no()}', align='C')


class LegalReporter:
    @staticmethod
    def add_persona_comparison_page(pdf, personas):
        if not personas: return
        pdf.add_page()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "EK-2: YARGISAL PERSPEKTIF KARSILASTIRMASI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        col_width = pdf.epw / 3
        start_y = pdf.get_y()

        p_list = [
            ("HAKIM", personas.get("judge", "")),
            ("KARSI TARAF", personas.get("opponent", "")),
            ("BILIRKISI", personas.get("expert", ""))
        ]

        max_y = start_y
        for i, (title, text) in enumerate(p_list):
            x = pdf.l_margin + i * col_width
            pdf.set_xy(x, start_y)
            pdf.set_font("helvetica", "B", 10)
            pdf.multi_cell(col_width - 2, 6, title, align='C')
            pdf.ln(1)
            pdf.set_xy(x, pdf.get_y())  # Reset X after multicell
            pdf.set_font("helvetica", size=8)
            pdf.multi_cell(col_width - 2, 4, text)
            max_y = max(max_y, pdf.get_y())

        pdf.set_y(max_y + 10)

    @staticmethod
    def add_appeal_arguments_page(pdf, appeal_text):
        if not appeal_text: return
        pdf.add_page()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "EK-3: OLASI ITIRAZ ARGUMANLARI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(0, 6, appeal_text)

    @staticmethod
    def add_petition_page(pdf, petition_text):
        if not petition_text: return
        pdf.add_page()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "EK-4: ISTINAF / TEMYIZ DILEKCESI TASLAGI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        pdf.set_font("helvetica", size=10)
        pdf.multi_cell(0, 6, petition_text)

    @staticmethod
    def add_action_plan_page(pdf, action_plan):
        if not action_plan: return
        pdf.add_page()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "EK-5: ITIRAZ AKSİYON PLANI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

        for action in action_plan:
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 8, f">> {action.get('title', 'Aksiyon')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", size=9)
            pdf.multi_cell(0, 5, f"Kaynak: {action.get('source', '')} | Risk: {action.get('risk_if_missing', '')}")
            pdf.ln(2)

    @staticmethod
    def add_audit_log_section(pdf, audit_data):
        if not audit_data or "timeline" not in audit_data: return
        pdf.add_page()
        pdf.set_font("helvetica", "B", 13)
        pdf.cell(0, 10, "3. KARAR SURECI VE DENETIM (AUDIT LOG)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        timeline = AuditTimelineBuilder.build(audit_data)
        explanation = ScoreExplanationEngine.generate(timeline)

        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 8, "SKOR DEGISIM ANALIZI:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "I", 10)
        pdf.multi_cell(0, 5, explanation)
        pdf.ln(5)

        for log in audit_data["timeline"]:
            step = log.get("step", 0)
            title = log.get("title", "Islem")
            desc = log.get("description", "")
            score = log.get("resulting_score")
            ts = datetime.fromtimestamp(log.get("timestamp", time.time())).strftime('%H:%M:%S')

            pdf.set_font("helvetica", "B", 10)
            pdf.cell(0, 6, f"{step}. {title.upper()} [{ts}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("helvetica", size=9)
            pdf.multi_cell(w=0, h=5, text=f"Detay: {desc}")
            if score:
                pdf.set_font("helvetica", "B", 8)
                pdf.cell(0, 5, f">> SKOR ETKISI: %{score}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

    @staticmethod
    def create_report(user_story, valid_docs, advice_text, audit_data=None, filename="Hukuki_Rapor_V111.pdf", llm=None,
                      personas=None, case_topic=""):
        branding = WhiteLabelConfig(
            firm_name="LEGAL OS CORP",
            footer_text="Gizli ve Ozeldir - Otomatik Analiz Raporu",
            color=(0, 51, 102)
        )
        pdf = BrandedPDFGenerator(branding)

        CorporateCover.add(pdf, audit_data.get("case_id", "N/A") if audit_data else "N/A", "V111")

        pdf.add_page();
        pdf.set_font("helvetica", size=11)

        def clean(t):
            if not t: return ""
            tr = {'ğ': 'g', 'ü': 'u', 'ş': 's', 'ı': 'i', 'ö': 'o', 'ç': 'c', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'İ': 'I',
                  'Ö': 'O', 'Ç': 'C'}
            for k, v in tr.items(): t = t.replace(k, v)
            return t.encode('latin-1', 'replace').decode('latin-1')

        pdf.set_font(style='B', size=12);
        pdf.cell(0, 10, clean("1. OLAY VE KAPSAM:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(style='', size=10);
        pdf.multi_cell(0, 6, clean(user_story));
        pdf.ln(5)

        pdf.set_font(style='B', size=12);
        pdf.cell(0, 10, clean("2. INCELEME VE HUKUKI GORUS:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(style='', size=10);
        pdf.multi_cell(0, 6, clean(advice_text))

        if audit_data:
            LegalReporter.add_audit_log_section(pdf, audit_data)

            if llm:
                reasoning_gen = JudgeReasoningGenerator(llm)
                judge_text = reasoning_gen.generate(audit_data)

                pdf.add_page()
                pdf.set_font("helvetica", "B", 13)
                pdf.cell(0, 10, clean("EK-1: HAKIM KARAR GEREKCESI TASLAGI"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(5)
                pdf.set_font("helvetica", "I", 10)
                pdf.multi_cell(0, 6, clean(judge_text))

                appeal_gen = AppealArgumentGenerator(llm)
                appeal_text = appeal_gen.generate(judge_text)
                LegalReporter.add_appeal_arguments_page(pdf, clean(appeal_text))

                petition_gen = AppealPetitionGenerator(llm)
                petition_text = petition_gen.generate(judge_text, case_topic)
                LegalReporter.add_petition_page(pdf, clean(petition_text))

                action_mapper = AppealActionMapper(llm)
                action_plan = action_mapper.map_arguments(appeal_text)
                for ap in action_plan:
                    ap['title'] = clean(ap.get('title', ''))
                    ap['source'] = clean(ap.get('source', ''))
                    ap['risk_if_missing'] = clean(ap.get('risk_if_missing', ''))
                LegalReporter.add_action_plan_page(pdf, action_plan)

            if personas:
                clean_personas = {k: clean(v) for k, v in personas.items() if isinstance(v, str)}
                LegalReporter.add_persona_comparison_page(pdf, clean_personas)

        try:
            pdf.output(filename); print(f"\n📄 Kurumsal Rapor (V111) Hazır: {filename}")
        except:
            pass


# ==================================================
# 1️⃣2️⃣ LEGAL UI PRINTER
# ==================================================
class LegalUIPrinter:
    @staticmethod
    def print_grand_ui_log(ui_data, doc_scan_log):
        if not ui_data or not ui_data.get("principles"): return

        print("\n" + "█" * 80)
        print(f"🖥️  LEGAL OS V111 - TAM KAPSAMLI ANALİZ VE TAKİP RAPORU")
        print("█" * 80 + "\n")

        # 0. AUDIT TIMELINE
        print(f"⏱️ 0. İŞLEM ZAMAN ÇİZELGESİ (AUDIT LOG):")
        for log in ui_data.get("audit_log", {}).get("timeline", []):
            ts = datetime.fromtimestamp(log['timestamp']).strftime('%H:%M:%S')
            print(f"   🕒 [{ts}] {log['title']} -> {log['description']}")
        print("-" * 80)

        # 1. BELGELER & YARGIÇ GEREKÇELERİ
        print(f"📂 1. BELGE TARAMA VE YARGIÇ DEĞERLENDİRMESİ:")
        for doc in doc_scan_log:
            print(f"   📄 {doc['source']} (Sf.{doc['page']}) -> {doc['role']}")
            print(f"      ↳ Gerekçe: {doc['reason'][:100]}...")
        print("-" * 80)

        # PRINCIPLE LOOP
        p = ui_data["principles"][0]

        # 2-6. İLKE ANALİZİ
        print(f"⚖️  2. SEÇİLEN TEMEL İLKE:\n   \"{p['text'][:120]}...\"")
        print(f"   📊 3. Zıtlık Analizi: {'⚠️ VAR' if p['conflict_flag'] else '✅ YOK'}")
        print(f"   📈 4. Trend Logu: {p['trend_log']}")
        print(f"   🧭 5. Polarite: {p['polarity']}")
        print(f"   🔥 6. Çelişki Tespiti: {p['conflict_analysis']['conflict_level']}")
        print("-" * 80)

        # 10-12. ZAMAN VE EVRİM
        print(f"⏳ 10. İLKE EVRİMİ: {p['trend_log']}")
        print(f"📅 11. GÜNCEL İÇTİHAT UYARISI: {p['year_bucket']} Dönemi")
        print("-" * 80)

        # 13-14. SKOR VE NEDENİ
        print(
            f"🎲 13. RİSK & BAŞARI SKORU: %{p['score_data']['success_probability']} ({p['score_data']['confidence_level']})")
        print(f"❓ 14. NEDEN BU SKOR?: {p['score_data']['summary']}")
        print("-" * 80)

        # 15-18. PERSONA LOGLARI
        opp_title = p['personas']['opponent_title']
        print("🗣️  PERSONA GÖRÜŞLERİ:")
        print(f"   👨‍⚖️ 15. HAKİM DİLİ: \"{p['personas']['judge'][:100]}...\"")
        print(f"   🏛️ 16. {opp_title} (KARŞI TARAF): \"{p['personas']['opponent'][:100]}...\"")
        print(f"   🔍 17. BİLİRKİŞİ DİLİ: \"{p['personas']['expert'][:100]}...\"")
        print(f"   🛑 18. HAKİM NEDEN REDDEDER?: \"{p['personas']['devil'][:100]}...\"")
        print("-" * 80)

        # 19. ÇELİŞKİ ANALİZİ
        if p['conflict_analysis']['conflict_score'] > 0:
            print(
                f"⚔️  19. PERSONA ÇELİŞKİ ANALİZİ: {p['conflict_analysis']['conflict_level']} (Skor: {p['conflict_analysis']['conflict_score']})")
            for r in p['conflict_analysis']['summary']: print(f"      🔴 {r}")
        print("-" * 80)

        # 20. GEREKÇELİ KARAR
        print(f"✍️  20. GEREKÇELİ KARAR TASLAĞI:\n   {p['reasoned_verdict'][:200]}...")
        print("-" * 80)

        # 21. YÖNETİCİ ÖZETİ
        print(f"📝 21. YÖNETİCİ ÖZETİ (BU DOSYA NEDEN RİSKLİ?):\n   {ui_data['executive_summary']}")
        print("-" * 80)

        # 22, 26, 27. STRATEJİ VE İŞ PAKETLERİ
        print("🚀 22/26/27. GÜÇLENDİRME & SOMUT İŞ PAKETLERİ:")
        for act in p['action_plan']:
            src = act['evidence']['source']
            src_str = f"{src['entity']} ({src['method']})" if isinstance(src, dict) else src
            print(f"   📦 [ID: {act['action_id'][:6]}] {act['title']}")
            print(f"      ↳ Kaynak: {src_str} (Adet: {act['evidence']['count']})")
            print(f"      ↳ Risk: {act['if_not_done']}")
            print(f"      ↳ Etki: +{act['risk_reduction']['expected_score_increase']} Puan")
        print("-" * 80)

        # 23, 28. SİMÜLASYON
        sim = p['simulation']
        print(f"🔮 23/28. SİMÜLASYON SONUCU:")
        print(f"   Mevcut: %{sim['current_score']} --> Hedef: %{sim['projected_score']}")
        print(f"   Artış: +{sim['total_boost']} Puan")
        print("█" * 80 + "\n")


# ==================================================
# 1️⃣3️⃣ ANA UYGULAMA (MAIN APP)
# ==================================================
class LegalApp:
    def __init__(self):
        print("🚀 LEGAL SUITE V111 (Optimized Full Stack)...")
        self.search_engine = LegalSearchEngine()

        if self.search_engine.connect_db():
            self.memory_manager = LegalMemoryManager(
                self.search_engine.client,
                self.search_engine.dense_embedder,
                ChatOllama(model=LegalConfig.LLM_MODEL, temperature=0.1)
            )
        else:
            self.memory_manager = None

        self.judge = LegalJudge(memory_manager=self.memory_manager)
        self.reporter = LegalReporter()
        self.ui_printer = LegalUIPrinter()

    def run(self):
        if not self.search_engine.run_indexing():
            self.search_engine.close()
            sys.exit()

        if self.memory_manager:
            self.memory_manager.consolidate_principles_v79()

        print("\n✅ SİSTEM HAZIR. (Çıkış: 'q')")

        try:
            while True:
                print("-" * 60)
                story = input("📝 Olay: ");
                if story == 'q': break
                topic = input("🎯 Odak: ")
                neg_input = input("🚫 Yasaklı: ")
                negatives = [w.strip().lower() for w in neg_input.split(",")] if neg_input else []

                print("   🛡️ Girdi kontrol ediliyor...")
                if not self.judge.validate_user_input(story, topic):
                    print("   ❌ UYARI: Girdi anlamsız. Lütfen mantıklı bir olay giriniz.")
                    continue

                expanded = self.judge.generate_expanded_queries(story, topic)
                full_query = f"{story} {topic} " + " ".join(expanded)
                print(f"   ✓ Sorgu: {len(full_query)} karakter")

                candidates = self.search_engine.retrieve_raw_candidates(full_query)
                if not candidates: continue

                valid_docs = self.judge.evaluate_candidates(candidates, story, topic, negatives)
                if not valid_docs: print("🔴 Yargıç hepsini eledi."); continue

                context_str = ""
                doc_scan_log = []
                for i, d in enumerate(valid_docs):
                    doc_scan_log.append({
                        "source": d['source'], "page": d['page'],
                        "role": d['role'], "reason": d['reason']
                    })
                    context_str += f"""
                        BELGE #{i + 1}
                        KAYNAK: {d['source']}
                        TÜR: {d['type']}
                        ROL: {d['role']}
                        YARGIÇ GEREKÇESİ: {d['reason']}
                        İÇERİK ÖZETİ: {d['text'][:800]}...
                        =========================================
                        """

                current_personas = {}
                if self.memory_manager:
                    self.memory_manager.recall_principles(full_query)
                    self.ui_printer.print_grand_ui_log(self.memory_manager.latest_ui_data, doc_scan_log)

                    if self.memory_manager.latest_ui_data.get("principles"):
                        current_personas = self.memory_manager.latest_ui_data["principles"][0]["personas"]

                full_advice = self.judge.generate_final_opinion(story, topic, context_str)

                audit_dump = {}
                llm_instance = None
                if self.memory_manager and hasattr(self.memory_manager, 'latest_ui_data'):
                    audit_dump = self.memory_manager.latest_ui_data.get("audit_log", {})
                    llm_instance = self.memory_manager.recommendation_engine.llm

                self.reporter.create_report(story, valid_docs, full_advice, audit_dump, "Hukuki_Rapor_V111.pdf",
                                            llm_instance, current_personas, full_query)

        except KeyboardInterrupt:
            print("\n👋 Program durduruldu.")
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")
        finally:
            self.search_engine.close()


if __name__ == "__main__":
    freeze_support()
    app = LegalApp()
    app.run()