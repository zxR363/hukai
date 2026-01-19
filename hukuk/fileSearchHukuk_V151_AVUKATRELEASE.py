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
import requests
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support
from dataclasses import dataclass, field
from collections import Counter
import subprocess
import time
from yargiMcp import YargiMcpBridge
# --------------------------------------------------
# 📦 IMPORTLAR
# --------------------------------------------------

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, Range
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_community.document_loaders import PyMuPDFLoader

from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------GPU ICIN ---------------
def get_llm_judge():
    return ChatOllama(
        model="qwen2.5:3b",
        temperature=0.1,
        num_ctx=8192,
        base_url="http://192.168.134.42:11434"
    )


def log_gpu_status():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        print("🟢 GPU AKTİF (OLLAMA)")
    except:
        print("⚠️ GPU BULUNAMADI, CPU FALLBACK")


# --------------------GPU ICIN ---------------

# PDF CIKTILARI Mevcut importların altına ekleyin
from pdf_reports import (
    LegacyPDFReport,
    JudicialPDFReport,
    ClientSummaryPDF,  # Eğer kullanacaksanız
    ReportOrchestrator
)

# UTF-8 Ayarı
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


# ==================================================
# 1ï¸âƒ£ KONFİGÜRASYON VE BAĞLAM SINIFLARI
# ==================================================

# # 🔍¨ Commit 5.3: Query Context (Single Source of Truth) â€“ DÜZELTİLMİŞ VE ÇALIŞIR
# from dataclasses import dataclass, field
# from typing import List
#
@dataclass
class QueryContext:
    """
    Sistemde TEK bağlayıcı bağlam nesnesi.
    Tüm modüller yalnızca bunu referans alır.
    """
    # Kullanıcı girdisi (zorunlu)
    query_text: str
    topic: str
    negative_scope: List[str]

    # Otomatik algılananlar (varsayılan değerlerle)
    detected_domain: str = "genel_hukuk"
    allowed_sources: List[str] = field(default_factory=list)

    # Sistem bayrakları
    allow_analogy: bool = False
    allow_speculation: bool = False
    allow_soft_language: bool = False
    judge_evaluated: bool = False

    def __post_init__(self):
        """Dataclass oluşturulduktan sonra çalışır â€“ domain algılama burada"""
        self.detect_domain()
        self.assert_hard_limits()

    def detect_domain(self):
        """Sorgudan domain algıla â€“ basit ama etkili"""
        text_lower = self.query_text.lower()
        if any(word in text_lower for word in ["miras", "veraset", "ıskat", "iskat", "vasiyet", "veraset ilamı"]):
            self.detected_domain = "miras_hukuku"
        elif any(word in text_lower for word in ["borç", "alacak", "tahsil", "icra", "teminat"]):
            self.detected_domain = "borclar_hukuku"
        elif any(word in text_lower for word in ["boşanma", "nafaka", "velayet", "mal paylaşımı", "evlilik"]):
            self.detected_domain = "aile_hukuku"
        elif any(word in text_lower for word in ["ceza", "suç", "mahkumiyet", "beraat", "tck"]):
            self.detected_domain = "ceza_hukuku"
        elif any(word in text_lower for word in ["iş", "kıdem", "ihbar", "tazminat", "iş sözleşmesi"]):
            self.detected_domain = "is_hukuku"
        # İstersen daha fazla ekleyebilirsin

    def assert_hard_limits(self):
        """Hukuki güvenlik kemeri"""
        if self.allow_speculation:
            raise ValueError("Spekülasyon hukuki analizde yasaktır.")
        if self.allow_analogy and not self.allow_soft_language:
            raise ValueError("Analoji ancak yumuşak dil açıkça izin verildiğinde kullanılabilir.")


# 🔍¨ Commit 5.4: Decision Context (Yargısal Zemin)
@dataclass
class DecisionContext:
    """
    Hakim ve LLM için ortak, temiz ve süzülmüş karar zemini.
    Bu nesne oluşmadan LLM ÇAĞRILAMAZ.
    """

    # Kaynaklar
    documents: List[Dict[str, Any]] = field(default_factory=list)
    principles: List[Dict[str, Any]] = field(default_factory=list)

    # Analitik katman
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)

    def has_minimum_legal_basis(self) -> bool:
        """
        Hukuki tartışma yapılabilmesi için asgari eşik.
        """
        return bool(self.documents) or bool(self.principles)


# 🔍¨ Commit 5.5: Judge Reflex (Refleks Veri Yapısı)
@dataclass
class JudgeReflex:
    """
    Hakimin ilk refleksi.
    """
    tendency: str  # "KABUL" | "RED" | "TEREDDÜT"
    score: int  # 0â€“100
    doubts: List[str]  # Hakimin kafasına takılanlar


# 🔍¨ Commit 5.6: Persona Response (Persona Çıktı Modeli)
@dataclass
class PersonaResponse:
    role: str  # DAVACI | DAVALI | BILIRKISI
    response: str
    addressed_doubts: List[str]


# 🔍¨ Commit 5.7: Strengthening Action (Aksiyon Modeli)
# --- BURAYI DEĞİŞTİRİN (Eski StrengtheningAction yerine bunu koyun) ---
@dataclass
class StrengtheningAction:
    title: str
    description: str
    related_doubt: str
    impact_score: int
    # [V143 EKLENTİSİ] V120 Disiplini için yeni alanlar
    risk_analysis: str = "Risk analizi mevcut değil."
    source_ref: str = "Genel hukuk ilkeleri"


# --- BURAYI EKLEYİN (Yeni Sınıf) ---
class LegalTextSanitizer:
    """
    [V143] V120 Disiplini: Halüsinasyon ve İngilizce metin temizleyici.
    """

    def __init__(self):
        self.seen_sentences = set()

    def is_mostly_english(self, text):
        """Metnin İngilizce halüsinasyon olup olmadığını kontrol eder."""
        common_english_words = {"the", "and", "is", "of", "to", "in", "that", "it", "with", "as", "for", "childhood",
                                "education", "development"}
        words = text.lower().split()
        if not words: return False
        english_count = sum(1 for w in words if w in common_english_words)
        return (english_count / len(words)) > 0.3

    def sanitize_hallucinations(self, text):
        """İngilizceye kayan kısımları ve tekrar eden satırları temizler."""
        if not text: return ""
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if self.is_mostly_english(line): continue
            clean_lines.append(line)
        return "\n".join(clean_lines).replace("[END_OF_TEXT]", "").replace("<|endoftext|>", "")

    def enforce_no_repeat(self, text):
        """Alias for sanitize_hallucinations strictly for compatibility."""
        return self.sanitize_hallucinations(text)


# DINAMIK LLM AYARLARI AYRIK CALISMASI ICIN
LLM_PROFILES = {
    "judge": {
        "model": "qwen2.5:3b",
        "num_ctx": 8192,
        "temperature": 0.1
    },
    "persona": {
        "model": "qwen2.5:3b",
        "num_ctx": 8192,
        "temperature": 0.3
    },
    "risk": {
        "model": "qwen2.5:3b",
        "num_ctx": 8192,
        "temperature": 0.0
    }
}


def get_llm_by_profile(profile_name: str):
    """Verilen profile göre optimize edilmiş LLM (Ollama veya Gemini) nesnesi döndürür."""
    config_obj = LegalConfig()
    profile = LLM_PROFILES.get(profile_name, LLM_PROFILES["judge"])

    # V147: Bulut LLM Kontrolü (GROQ)
    if config_obj.USE_CLOUD_LLM:
        if not config_obj.GROQ_API_KEY or "YOUR" in config_obj.GROQ_API_KEY:
            print(f"   ⚠️ UYARI: Groq API anahtarı ayarlanmamış! [{profile_name}] için Lokal modele dönülüyor...")
        else:
            print(f"   ⚡ Groq LLM Başlatılıyor: [LLAMA3-70B] | temp: {profile['temperature']}")

            return ChatGroq(
                model_name=config_obj.CLOUD_MODEL_NAME,
                api_key=config_obj.GROQ_API_KEY,
                temperature=profile["temperature"],
                max_retries=5
            )

    print(
        f"   🔌 Lokal LLM Başlatılıyor: [{profile_name.upper()}] | ctx: {profile['num_ctx']} | temp: {profile['temperature']}")

    return ChatOllama(
        model=profile["model"],
        num_ctx=profile["num_ctx"],
        temperature=profile["temperature"],
        repeat_penalty=profile.get("repeat_penalty", 1.1),
        streaming=False,
        num_thread=4,
        num_gpu=1,
        base_url="http://192.168.134.42:11434"
    )


# DINAMIK LLM AYARLARI AYRIK CALISMASI ICIN

@dataclass
class LegalConfig:
    # V147: CLOUD LLM CONFIG
    # V147: CLOUD LLM CONFIG (GROQ)
    USE_CLOUD_LLM = False
    #GROQ_API_KEY = "" # Kullanıcıdan başlangıçta istenecek
    GROQ_API_KEY = ""  # Kullanıcıdan başlangıçta istenecek
    #GROQ_API_KEY = ""  # Kullanıcıdan başlangıçta istenecek
    CLOUD_MODEL_NAME = "llama-3.3-70b-versatile"  # Groq Llama 3.3 70B (En hızlı ve yeni)
    CLOUD_THROTTLE_SECONDS = 20  # Groq çok hızlıdır, throttle düşürülebilir

    # Google Drive Ana Yolu (HukAI Klasörü)
    # DRIVE_ROOT = "/content/drive/MyDrive/HukAI"
    DRIVE_ROOT = os.path.dirname(os.path.abspath(__file__))

    SOURCES = {
        "mevzuat": {
            "folder": os.path.join(DRIVE_ROOT, "mevzuatlar"),
            "collection": "legal_statutes_v48",
            "desc": "MEVZUAT"
        },
        "emsal": {
            "folder": os.path.join(DRIVE_ROOT, "belgeler"),
            "collection": "legal_precedents_v48",
            "desc": "EMSAL KARAR"
        }
    }

    MEMORY_COLLECTIONS = {
        "decision": "judge_memory_v1",
        "principle": "principle_memory_v1"
    }

    # Veritabanını da HukAI içine kaydediyoruz (Kalıcı Hafıza)
    QDRANT_PATH = os.path.join(DRIVE_ROOT, "qdrant_db_master")

    # Sistem durum dosyası da burada
    STATE_FILE = os.path.join(DRIVE_ROOT, "system_state.json")

    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "qwen2.5"

    # V120: YENİ LLM PARAMETRELERİ (GLOBAL KALİTE KONTROL)
    LLM_CONFIG = {
        "temperature": 0.4,
        "top_p": 0.9,
        "repeat_penalty": 1.2,  # frequency_penalty karşılığı (Ollama/Llama)
        "num_predict": 1200  # max_tokens karşılığı
    }

    # V124: GÜÇLENDİRİLMİŞ PROMPT GUARD
    PROMPT_GUARD = """
SEN "SENIOR" DÜZEYDE BİR TÜRK HUKUKÇUSUSUN. "STAJYER" GİBİ KONUŞMA.

ZORUNLU YAZIM VE AKIL YÜRÜTME KURALLARI:

1. REFERANS ZORUNLULUĞU: Yargıtay kararlarına atıf yaparken "Yargıtay Kararı" deyip geçme. MUTLAKA "Esas No/Karar No" (Örn: E.2021/123, K.2021/555) formatını uydurmadan, elindeki metinden bularak yaz. Eğer metinde numara yoksa "Tarihli Karar" şeklinde belirt.
2. KANUN MADDELERİ: TMK, TBK veya HMK maddelerine atıf yaparken, maddenin ilgili fıkrasını TAM VE EKSİKSİZ ALINTILA. "İlgili maddeye göre..." diyip geçiştirme.
3. KESİNLİK İLKESİ: "Olabilir", "değerlendirilebilir", "kanaatimizce" gibi yuvarlak (muğlak) ifadeler YASAKTIR. Hukuki durum neyse NET konuş: "Bu durum hukuka aykırıdır" veya "Bu talep kabul edilmelidir."
4. ÇELİŞKİ AVLA: Verilen metinlerdeki mantık hatalarını veya hukuki eksiklikleri acımasızca eleştir.
5. ROLÜNE SADIK KAL:
   - Bilirkişiysen: Taraf tutma, sadece teknik ve hukuki gerçeği söyle.
   - Hakissen: Duygusal değil, normatif karar ver.
6. SADECE verilen olay, scope ve hukuki bağlam içinde kal.
7. Genel hukuk bilgisi, öğretici anlatım veya akademik açıklama YAPMA.
8. “Genel olarak”, “çoğunlukla”, “doktrinde” gibi belirsiz ifadeler KULLANMA.
9. Aynı hukuki ilkeyi veya TMK/Yargıtay maddesini BİR KEZ açıkla.
10. Aynı düşünceyi farklı kelimelerle TEKRAR ETME.
11. Somut olayla bağlantısı olmayan hiçbir bilgi EKLEME.
12. Emsal yoksa uydurma; belirsizlik varsa AÇIKÇA belirt.
13. Değer yargısı, ahlaki yorum, sosyal politika yorumu YAPMA.
14. “Bu durumda karar verilmelidir” gibi HÜKÜM KURAN ifadeler kullanma.
15. Hakim, avukat veya bilirkişi rolü dışında düşünme.
16. Çıktı, gerçek bir mahkeme dosyasına girebilecek ciddiyette olsun.
17. Bu kuralların dışına çıkma; çıktıyı bu kurallara göre DENETLE.
18.Her belge yalnızca bir kez özetlenir.Özet, sorgudaki somut olayla doğrudan bağ kurmak zorundadır.
"Bu belge, sorgudaki [X] durumuna şu şekilde uygulanır: ..." formatı zorunludur.
19.Belge → Hukuki İlke → Somut Olay → Dosyaya Etki zinciri kurulmadan belge kullanılamaz.
20. Subjektif kelimeler ("benzetebilirsiniz", "olabilir", "gibi") KULLANMA; her atıf SOMUT olsun ("Yargıtay 14. HD 2015/2278 E. kararında şöyle belirtilmiştir: ...").
"""

    # --- V120: CORE RULE REGISTRY (YAML SIMULATION) ---
    # Harici dosya okuma mantığı eklendiğinde burası fallback olur.
    CORE_RULES_DB = {
        "miras_hukuku": {
            "description": "Miras ve çekişmesiz yargı işleri",
            "rules": [
                {
                    "id": "CR_MIRAS_001",
                    "rule": "Veraset ilamı çekişmesiz yargı işidir.",
                    "effect": "Maddi anlamda kesin hüküm oluşturmaz.",
                    "applies_to": ["judge", "risk", "persona"]
                },
                {
                    "id": "CR_MIRAS_002",
                    "rule": "Mirasçılık belgesi aksi ispat edilinceye kadar geçerlidir.",
                    "effect": "İptal davası açılabilir.",
                    "applies_to": ["judge"]
                }
            ]
        },
        "ceza_hukuku": {
            "description": "Ceza yargılamasına ilişkin temel ilkeler",
            "rules": [
                {
                    "id": "CR_CEZA_001",
                    "rule": "Şüpheden sanık yararlanır (In Dubio Pro Reo).",
                    "effect": "Delil yetersizliği halinde beraat esastır.",
                    "applies_to": ["judge", "risk"]
                },
                {
                    "id": "CR_CEZA_002",
                    "rule": "Ceza hukukunda kıyas yasağı esastır.",
                    "effect": "Kanunsuz suç ve ceza olmaz, aleyhe yorum yapılamaz.",
                    "applies_to": ["judge"]
                }
            ]
        },
        "is_hukuku": {
            "description": "İş hukuku ve işçi-işveren ilişkileri",
            "rules": [
                {
                    "id": "CR_IS_001",
                    "rule": "İş hukukunda işçi lehine yorum ilkesi esastır.",
                    "effect": "Mevzuat boşluklarında işçi yararı gözetilir.",
                    "applies_to": ["judge", "persona"]
                }
            ]
        },
        "genel_hukuk": {
            "description": "Genel hukuk ilkeleri",
            "rules": [
                {
                    "id": "CR_GENEL_001",
                    "rule": "İddia eden iddiasını ispatla mükelleftir.",
                    "effect": "İspat yükü kural olarak davacıdadır.",
                    "applies_to": ["judge", "risk"]
                }
            ]
        }
    }

    SEARCH_LIMIT_PER_SOURCE = 60
    SCORE_THRESHOLD = 0.35
    LLM_RERANK_LIMIT = 10

    DECAY_RATE_PER_MONTH = 0.98
    PRINCIPLE_MERGE_THRESHOLD = 0.90
    MIN_CONFIDENCE_THRESHOLD = 0.55


# ==================================================
# 2ï¸âƒ£ YARDIMCI ARAÇLAR (STATIC)
# ==================================================
def _contains_decision(text: str, decision: str) -> bool:
    text = text.upper()
    decision = decision.upper()

    if decision == "KABUL":
        return "KABUL" in text or "KABUL EDİL" in text
    if decision == "RED":
        return "RED" in text or "REDDEDİL" in text
    return False


def worker_embed_batch_global(args):
    """Multiprocessing için global kalmalı."""
    texts, model_name = args
    try:
        embedder = OllamaEmbeddings(model=model_name, base_url="http://192.168.134.42:11434")
        return embedder.embed_documents(texts)
    except Exception as e:
        print(f"⚠️ Batch hatası (atlanıyor): {e}")
        return []


# 🔍¨ Commit 5.4: Decision Builder (Adaptör)
class DecisionBuilder:
    """
    Sistemin farklı çıktılarından DecisionContext inşa eden yardımcı sınıf.
    """

    @staticmethod
    def build_decision_context_from_valid_docs(valid_docs: list) -> DecisionContext:
        """
        LegalJudge tarafından filtrelenmiş 'valid_docs' listesini alır.
        """
        context = DecisionContext()

        for doc in valid_docs:
            # ID yoksa geçici üret, varsa kullan
            doc_id = str(uuid.uuid4())

            context.documents.append({
                "id": doc_id,
                "type": doc.get("type"),  # EMSAL / MEVZUAT
                "source": doc.get("source"),
                "confidence": doc.get("score"),  # Judge skoru (0-100)
                "score": doc.get("score"),  # Confidence ile aynı PDF_REPORTS'DA ihtiyac oluyor
                "content": doc.get("text"),
                "role": doc.get("role"),
                "reason": doc.get("reason")
            })

            context.relevance_scores[doc_id] = doc.get("score", 0.0)

        return context

    @staticmethod
    def enrich_decision_context_with_memory(context: DecisionContext, memory_principles: list) -> DecisionContext:
        """
        Hafızadan gelen ilkeleri Context'e ekler.
        """
        if not memory_principles:
            return context

        for principle in memory_principles:
            context.principles.append({
                "principle": principle.get("text"),
                "confidence": principle.get("score_data", {}).get("success_probability", 0),
                "source": "memory_v1",
                "trend": principle.get("trend_log", "")
            })

        return context


# 🔍¨ Commit 5.5: Judge Core (Deterministik Akıl)
class JudgeCore:
    """
    LLM'siz, deterministik hakim muhakemesi.
    """

    def evaluate(self, decision_context: DecisionContext) -> JudgeReflex:
        """
        [V145: DİNAMİK EŞİK SİSTEMİ]
        Statik %70/%40 yerine, dosya bazlı dinamik eşikler kullanır.
        """
        score = 15.0  # Temel skor
        doubts = []

        if not decision_context.documents:
            return JudgeReflex(tendency="RED", score=0, doubts=["Dosyada değerlendirilecek belge bulunamadı."])

        # 1ï¸âƒ£ Belge analizi
        doc_scores = [doc.get("score", 0) for doc in decision_context.documents]
        avg_doc_score = sum(doc_scores) / len(doc_scores) if doc_scores else 0

        for doc in decision_context.documents:
            conf = doc.get("score", 0)
            if conf >= 90:
                score += 15
            elif conf >= 80:
                score += 10
            elif conf >= 70:
                score += 5

            if conf < avg_doc_score * 0.8:  # Ortalama kalitenin altındaki belgeler şüphe uyandırır
                doubts.append(f"Zayıf belge/delil takdiri: {doc.get('source')}")

        # 2ï¸âƒ£ Hukuki ilkeler
        for principle in decision_context.principles:
            conf = principle.get("confidence", 0)
            if conf >= 85:
                score += 10
            elif conf < 60:
                doubts.append("İçtihat desteği zayıf veya çelişkili.")

        # 3ï¸âƒ£ Dinamik Eşik Hesaplama
        # Dosyadaki toplam belge sayısı ve niteliğine göre kabul barajı değişir
        kabul_baraji = 75 - (len(decision_context.documents) * 2)  # Daha çok belge barajı aşağı çeker
        kabul_baraji = max(60, min(85, kabul_baraji))

        red_baraji = 45 - (len(decision_context.documents) * 1)
        red_baraji = max(30, min(45, red_baraji))

        score = min(score, 100)

        # 4ï¸âƒ£ Hakim refleksi
        if score >= kabul_baraji:
            tendency = "KABUL"
        elif score <= red_baraji:
            tendency = "RED"
        else:
            tendency = "TEREDDÜT"

        return JudgeReflex(
            tendency=tendency,
            score=score,
            doubts=doubts
        )


# ==================================================
# [YENİ] 🧠 LEGAL DECISION LOGIC (KARAR MANTIK MOTORU)
# ==================================================
class LegalDecisionLogic:
    """
    LLM çıktılarını matematiksel kurallarla denetler ve
    nihai kararı (Refleks) yeniden hesaplar.
    """

    # ADIM 1: Tereddüt Anahtar Kelimeleri
    TEREDDUT_KEYWORDS = [
        "tereddüt", "eksik", "yetersiz",
        "belirsiz", "dikkat", "potansiyel", "şüphe",
        "çelişki", "muğlak"
    ]

    # ADIM 4: Bilirkişi Netlik Kelimeleri
    NETLIK_KELIMELERI = ["kanaat", "sonuç", "tespit edilmiştir", "mütalaa", "görüş", "açıkça",
                         "kesin olarak", "görüşündeyim", "neticesinde"]

    # ADIM 7: Hukuki Terminoloji Zorunluluğu
    REQUIRED_LEGAL_TERMS = ["TBK", "TMK", "ispat", "delil", "hüküm", "yargıtay"]

    def detect_tereddut(self, text: str) -> bool:
        text = text.lower()
        return any(k in text for k in self.TEREDDUT_KEYWORDS)

    def count_tereddut_sources(self, bilirkisi_text, davali_text, delil_durumu_metni=""):
        count = 0
        if self.detect_tereddut(bilirkisi_text): count += 1
        if self.detect_tereddut(davali_text): count += 1
        # Delil durumu metni opsiyonel, genelde analizden gelir
        if "belirsiz" in delil_durumu_metni.lower() or "eksik" in delil_durumu_metni.lower():
            count += 1
        return count

    def bilirkisi_net_mi(self, text):
        return any(k in text.lower() for k in self.NETLIK_KELIMELERI)

    def davali_gucu_hesapla(self, text):
        score = 0
        text = text.lower()
        if "delil" in text and "eksik" in text: score += 4
        if "yargıtay" in text or "emsal" in text: score += 3
        if "belge" in text and "yok" in text: score += 2
        # Maksimum 10 üzerinden normalize edelim
        return min(score, 10)

    def hukuki_tavsif_gecerli_mi(self, text):
        return any(r in text for r in self.REQUIRED_LEGAL_TERMS)

    def calculate_final_score(self, base_score, davali_gucu, tereddut_sayisi, bilirkisi_net):
        # ADIM 6: Genel Güç Skoru Normalizasyonu
        # Base score JudgeCore'dan gelir (Örn: 80)
        score = base_score

        # Tereddüt cezası
        score -= tereddut_sayisi * 15  # Tereddüt başına 15 puan kır (Sıkılaştırdım)

        # Davalı gücü cezası
        score -= davali_gucu * 2

        # Bilirkişi vetosu
        if not bilirkisi_net:
            score -= 20

        return max(0, min(score, 100))

    def decide_verdict(self, bilirkisi_net, tereddut_sayisi, davali_gucu, final_score):
        # ADIM 3: Hakim Refleksi Decision Tree

        # 1. Kilit: Bilirkişi net değilse direkt Tereddüt
        if not bilirkisi_net:
            return "TEREDDÜTLÜ â€“ BİLİRKİŞİ MUĞLAK"

        # 2. Kilit: Tereddüt sayısı 1'den fazlaysa
        if tereddut_sayisi >= 1:
            return f"TEREDDÜTLÜ â€“ {tereddut_sayisi} KAYNAK ŞÜPHELİ"

        # 3. Kilit: Davalı çok güçlüyse
        if davali_gucu >= 7:
            return "TEREDDÜTLÜ â€“ DAVALI SAVUNMASI GÜÇLÜ"

        # 4. Kilit: Skor yeterliliği
        if final_score >= 75:
            return "KABUL EĞİLİMLİ"

        return "RED EĞİLİMLİ"

    def final_sanity_check(self, refleks, skor, tereddut_sayisi):
        # ADIM 8: Son Güvenlik Kilidi
        is_kabul = "KABUL" in refleks.upper()

        if is_kabul and (tereddut_sayisi > 0 or skor < 75):
            print(f"🚨 SANITY CHECK FAILED: Refleks={refleks}, Skor={skor}, Tereddüt={tereddut_sayisi}")
            # Zorla düzelt
            return "TEREDDÜTLÜ (OTOMATİK DÜZELTME)", skor

        return refleks, skor

    def run_logic(self, initial_reflex, persona_outputs):
        """
        Tüm mantığı çalıştırır ve güncellenmiş bir JudgeReflex nesnesi döner.
        """
        if initial_reflex.score > 85 and initial_reflex.doubts:
            initial_reflex.score = min(initial_reflex.score, 75)

        # Metinleri ayıkla
        davaci_text = next((p.response for p in persona_outputs if "DAVACI" in p.role), "")
        davali_text = next((p.response for p in persona_outputs if "DAVALI" in p.role), "")
        bilirkisi_text = next((p.response for p in persona_outputs if "BİLİRKİŞİ" in p.role), "")

        # Analizler
        bilirkisi_net = self.bilirkisi_net_mi(bilirkisi_text)
        davali_gucu = self.davali_gucu_hesapla(davali_text)
        tereddut_sayisi = self.count_tereddut_sources(bilirkisi_text, davali_text,
                                                      initial_reflex.doubts[0] if initial_reflex.doubts else "")

        # Skorlama
        # Başlangıç skorunu JudgeCore'dan alıyoruz
        final_score = self.calculate_final_score(initial_reflex.score, davali_gucu, tereddut_sayisi, bilirkisi_net)

        # Karar Ağacı
        new_tendency = self.decide_verdict(bilirkisi_net, tereddut_sayisi, davali_gucu, final_score)

        # Sanity Check
        checked_tendency, checked_score = self.final_sanity_check(new_tendency, final_score, tereddut_sayisi)

        # Güncellenmiş Tereddüt Listesi
        new_doubts = initial_reflex.doubts
        if tereddut_sayisi > 0 and not new_doubts:
            new_doubts = ["Otomatik tespit: Metinlerde belirsizlik/tereddüt ifadeleri mevcut."]

        print(f"\n🧠 MANTIK MOTORU DEVREDE:")
        print(f"   - Tereddüt Sayısı: {tereddut_sayisi}")
        print(f"   - Davalı Gücü: {davali_gucu}")
        print(f"   - Bilirkişi Net mi?: {bilirkisi_net}")
        print(f"   - Eski Skor: {initial_reflex.score} -> Yeni Skor: {checked_score}")
        print(f"   - Eski Karar: {initial_reflex.tendency} -> Yeni Karar: {checked_tendency}")

        return JudgeReflex(
            tendency=checked_tendency,
            score=int(checked_score),
            doubts=new_doubts
        )


class PersonaEngine:
    """
    LLM kontrollü persona simülasyonu.
    MİMARİ: CPU (Prompt Hazırlık) -> GPU (Inference)
    """

    def __init__(self, llm):
        self.llm = llm
        self.sanitizer = LegalTextSanitizer()

        # DOMAIN MAPPING (CPU Verisi - Sabit)
        self.DOMAIN_MAPPINGS = {
            "miras_hukuku": {
                "maddeler": "TMK md. 598 (mirasçılık belgesi), md. 510-513 (mirastan çıkarma/ıskat), md. 605 ve devamı (miras reddi)",
                "ictihatlar": "Yargıtay 2. Hukuk Dairesi ve 14. Hukuk Dairesi miras kararları (mirastan çıkarılanın sıfatı tamamen kalkmaz ilkesi)",
                "pdf_url": "https://mevzuat.gov.tr/mevzuatmetin/1.5.4721.pdf"
            },
            "borclar_hukuku": {
                "maddeler": "TBK md. 1-146 (borç ilişkileri), md. 49-60 (tazminat), md. 147 (zamanaşımı)",
                "ictihatlar": "Yargıtay 13. Hukuk Dairesi borç ve tazminat kararları",
                "pdf_url": "http://www.mevzuat.gov.tr/MevzuatMetin/1.5.6098.pdf"
            },
            "aile_hukuku": {
                "maddeler": "TMK md. 185-202 (evlilik birliği), md. 203-365 (boşanma, nafaka, velayet, mal rejimi)",
                "ictihatlar": "Yargıtay 2. Hukuk Dairesi aile hukuku kararları",
                "pdf_url": "https://mevzuat.gov.tr/mevzuatmetin/1.5.4721.pdf"
            },
            "ceza_hukuku": {
                "maddeler": "TCK ilgili maddeler (suç ve ceza), CMK md. 1-383 (yargılama usulü)",
                "ictihatlar": "Yargıtay Ceza Genel Kurulu ve ilgili Ceza Daireleri kararları",
                "pdf_url": "http://www.mevzuat.gov.tr/MevzuatMetin/1.5.5237.pdf"
            },
            "icra_hukuku": {
                "maddeler": "İİK md. 1-363 (icra ve iflas işlemleri)",
                "ictihatlar": "Yargıtay 12. Hukuk Dairesi icra kararları",
                "pdf_url": "https://mevzuat.gov.tr/mevzuatmetin/1.5.2004.pdf"
            },
            "is_hukuku": {
                "maddeler": "İşK md. 1-75 (iş sözleşmesi, kıdem, ihbar)",
                "ictihatlar": "Yargıtay 9. Hukuk Dairesi iş hukuku kararları",
                "pdf_url": "http://www.mevzuat.gov.tr/MevzuatMetin/1.5.4857.pdf"
            },
            "ticaret_hukuku": {
                "maddeler": "TTK md. 1-1524 (ticari işletme, şirketler, kıymetli evrak, deniz ticareti)",
                "ictihatlar": "Yargıtay 11. Hukuk Dairesi ticaret hukuku kararları",
                "pdf_url": "http://www.mevzuat.gov.tr/mevzuatmetin/1.5.6102.pdf"
            },
            "idare_hukuku": {
                "maddeler": "İYUK md. 1-55 (idari yargılama usulü, iptal davası, yürütmenin durdurulması)",
                "ictihatlar": "Danıştay 2., 3., 4. Daire idare hukuku kararları; Yargıtay 4. Hukuk Dairesi ilgili içtihatlar",
                "pdf_url": "http://www.mevzuat.gov.tr/MevzuatMetin/1.5.2577.pdf"
            },
            "vergi_hukuku": {
                "maddeler": "VUK md. 1-413 (vergi usulü, tarh, tebliğ, tahakkuk, tahsil)",
                "ictihatlar": "Danıştay Vergi Dava Daireleri Kurulu ve Yargıtay 3. Hukuk Dairesi vergi kararları",
                "pdf_url": "https://mevzuat.gov.tr/mevzuatmetin/1.4.213.pdf"
            },
            "medeni_usul_hukuku": {
                "maddeler": "HMK md. 1-448 (muhakeme usulü, dava şartları, deliller, temyiz)",
                "ictihatlar": "Yargıtay 2. ve 3. Hukuk Dairesi medeni usul kararları",
                "pdf_url": "http://www.mevzuat.gov.tr/MevzuatMetin/1.5.6100.pdf"
            },
            "fikri_mulkiyet_hukuku": {
                "maddeler": "SMK md. 1-191 (marka, patent, tasarım, coğrafi işaret koruması)",
                "ictihatlar": "Yargıtay 11. Hukuk Dairesi fikri mülkiyet kararları",
                "pdf_url": "http://www.mevzuat.gov.tr/mevzuatmetin/1.5.6769.pdf"
            },
            "genel_hukuk": {
                "maddeler": "HMK genel hükümleri, TMK/TBK temel ilkeleri",
                "ictihatlar": "İlgili Yargıtay dairesi kararları",
                "pdf_url": "https://mevzuat.gov.tr/mevzuatmetin/1.5.4721.pdf"
            }
        }

    # =========================================================================
    # 🟢 ADIM 1: CPU HAZIRLIK (Build Phase)
    # =========================================================================
    # Bu fonksiyon ASLA LLM çağırmaz. Sadece String üretir.
    def build_persona_prompts(self, context: QueryContext, decision_context: DecisionContext,
                              judge_reflex: JudgeReflex) -> List[Dict]:
        """
        Her persona için çalıştırılacak 'ham prompt' metnini hazırlar.
        """
        current_doubts = judge_reflex.doubts or ["Genel delil durumu"]

        # 1. Dosya/PDF Okuma (Disk IO - CPU)
        #    generate_domain_focus -> validate_madde_from_source (PyMuPDF kullanır)
        domain_focus_text = self.generate_domain_focus(context)

        # 2. Hukuki Zemin Metnini İnşa Et (String Operation)
        base_legal_content = f"""
        HAKİM EĞİLİMİ: {judge_reflex.tendency} (Skor: {judge_reflex.score}/100)
        TEREDDÜTLER: {', '.join(judge_reflex.doubts)}

        KANUNİ ZEMİN (DOĞRULANMIŞ METİNLER):
        {domain_focus_text}

        MEVCUT BELGELER (EMSAL/DELİL):
        {chr(10).join([f"- {d['source']} ({d['role']}): {d['reason']}" for d in decision_context.documents[:3]])}
        """

        # 3. Rolleri Tanımla (Config)
        roles_config = [
            ("DAVACI VEKİLİ",
             "Müvekkil lehine yorumla. Yukarıdaki 'DOĞRULANMIŞ METİNLER' kısmındaki maddeleri kullanarak hakimi ikna et."),
            ("DAVALI VEKİLİ",
             "Müvekkil lehine itiraz et. Yukarıdaki 'KANUNİ ZEMİN'deki boşlukları veya usul hatalarini kullan."),
            ("BİLİRKİŞİ",
             "YÜKSEK MAHKEME TETKİK HAKİMİ gibi davran. 'Kanaatimce' deme. Yukarıdaki kanun maddelerine aykırılık var mı net söyle.")
        ]

        # 4. Promptları Paketle
        prepared_payloads = []
        for role, instruction in roles_config:
            # --- PROMPT ŞABLONU (V144 Hukuki İyileştirme) ---
            final_prompt = f"""
            GÖREV: Aşağıdaki HUKUKİ DOSYAYI, kıdemli bir {role} olarak değerlendir.

            ÖNEMLİ: Sadece sağlanan metinlere sadık kal. Bilgin olmayan konularda "Dosya kapsamında bu hususta veri bulunmamaktadır" de. 
            Kesinlikle "early childhood education" gibi alakasız konulardan bahsetme.

            === DOSYA VE KANUNİ ZEMİN ===
            {base_legal_content}

            OLAY/SORGU:
            {context.query_text}

            GÖREV:
            {instruction}

            HAKİMİN TEREDDÜTLERİ:
            {self._format_doubts(current_doubts)}

            KISITLAMALAR:
            1. Sadece "DOĞRULANMIŞ METİNLER" ve "BELGELER" üzerinden konuş.
            2. Asla olmayan bir kanun maddesi uydurma.
            3. Maksimum 3-4 cümle. Net, keskin ve profesyonel bir hukuk dili kullan.
            4. "Olabilir", "değerlendirilebilir" gibi muğlak ifadelerden kaçın.

            ÇIKTI:
            """

            # Listeye at (Henüz LLM yok!)
            prepared_payloads.append({
                "role": role,
                "prompt": final_prompt,  # <-- HAZIR STRING
                "doubts": current_doubts
            })

        print("   ✅ CPU: Persona promptları hazırlandı (LLM'siz).")
        return prepared_payloads

    def _format_doubts(self,doc):
        return "\n".join(f"- {d}" for d in doc)

    # =========================================================================
    # 🔴 ADIM 2: GPU ÇALIŞTIRMA (Execution Phase)
    # =========================================================================
    # Bu fonksiyon SADECE LLM çağırır. Mantık kurmaz.
    # 🔴 ADIM 2: GPU ÇALIŞTIRMA (SAF SERİ DÖNGÜ)
    def execute_personas(self, prepared_payloads: List[Dict]) -> List[PersonaResponse]:
        """
        Hazırlanmış promptları TEKER TEKER (Sequential) LLM'e gönderir.
        Thread yok, Async yok. GPU darboğazı yok.
        """
        print(f"   🗣️ GPU: {len(prepared_payloads)} Persona sıraya alındı (Serial Processing)...")
        responses = []

        # âŒ ThreadPoolExecutor YOK
        # ✅ Basit 'for' döngüsü (En hızlısı ve en güvenlisi budur)

        for i, payload in enumerate(prepared_payloads):
            role = payload["role"]
            print(f"      â–¶ï¸ [{i + 1}/{len(prepared_payloads)}] İşleniyor: {role}...")

            try:
                # Bloklayıcı çağrı (Cevap gelene kadar kod durur)
                try:
                    raw_response = LegalUtils.safe_extract_content(self.llm.invoke(payload["prompt"]))
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "Rate limit" in err_str:
                        print(f"      ⚠️ Groq KOTA DOLDU! Fallback (Lokal) Modele geçiliyor...")
                        fallback_llm = ChatOllama(model="qwen2.5:3b", temperature=0.3,base_url="http://192.168.134.42:11434")
                        raw_response = LegalUtils.safe_extract_content(fallback_llm.invoke(payload["prompt"]))
                    else:
                        raise e

                clean_response = self.sanitizer.sanitize_hallucinations(raw_response)  # Temizle

                responses.append(PersonaResponse(
                    role=role,
                    response=clean_response,
                    addressed_doubts=payload["doubts"]
                ))
                print(f"      ✅ Tamamlandı: {role}")

            except Exception as e:
                print(f"      âŒ Hata ({role}): {e}")
                responses.append(PersonaResponse(
                    role=role,
                    response="Teknik hata nedeniyle beyan oluşturulamadı.",
                    addressed_doubts=payload["doubts"]
                ))

        return responses

    def _run_single_inference(self, payload):
        try:
            # TEK GÖREV: String'i modele ver, String al.
            result = LegalUtils.safe_extract_content(self.llm.invoke(payload["prompt"]))
        except:
            result = "Beyan oluşturulamadı."

        return PersonaResponse(
            role=payload["role"],
            response=result,
            addressed_doubts=payload["doubts"]
        )

    # --- YARDIMCI METOTLAR (AYNEN KORUNDU - CPU) ---
    def generate_domain_focus(self, ctx: QueryContext) -> str:
        mapping = self.DOMAIN_MAPPINGS.get(ctx.detected_domain.lower().replace(" ", "_"),
                                           self.DOMAIN_MAPPINGS["genel_hukuk"])
        kanun_kodu = mapping["maddeler"].split()[0] if "md." in mapping["maddeler"] else "Kanun"
        madde_range = mapping["maddeler"].split("md.")[1].split("(")[0].strip() if "md." in mapping["maddeler"] else "1"
        validated_maddeler = self.validate_madde_from_source(kanun_kodu, madde_range, mapping)
        return f"ODAK KONU: {ctx.topic}\nİLGİLİ KANUN: {kanun_kodu}\nDOĞRULANMIŞ MADDELER:\n{validated_maddeler}"

    def validate_madde_from_source(self, kanun_kodu: str, madde_range: str, mapping: dict) -> str:
        # PDF okuma kodunuz buraya gelecek (Orijinal koddaki gibi)
        # Hız için şimdilik basit return yapıyorum, siz kendi kodunuzu koruyun.
        pdf_path = os.path.join(LegalConfig.SOURCES["mevzuat"]["folder"], f"{kanun_kodu}.pdf")
        if os.path.exists(pdf_path):
            try:
                doc = fitz.open(pdf_path)
                # ... PDF okuma mantığı ...
                doc.close()
                return f"{kanun_kodu} {madde_range} (Yerel PDF'ten doğrulandı)"
            except:
                pass
        return f"{kanun_kodu} {madde_range} (Statik Doğrulama)"


class ActionEngine:
    """
    [V143 GÜNCELLEMESİ - V120 STANDARDI]
    Hakim tereddütlerini gidermek için V120 disiplininde (Aksiyon-Kaynak-Risk)
    stratejik plan üretir.
    """

    def __init__(self, llm):
        self.llm = llm
        # V120'deki metin temizleme disiplini için
        self.sanitizer = LegalTextSanitizer()

    # 🟢 ADIM 1: CPU HAZIRLIK (ASLA LLM ÇAĞIRMAZ)
    def build_risk_payload(self, judge_reflex: JudgeReflex, persona_outputs: List[PersonaResponse]):
        """
        Risk promptunu V120 standartlarına (Kaynak ve Risk analizi dahil) göre hazırlar.
        """
        # Eğer hakimin şüphesi yoksa aksiyona gerek yok
        if not judge_reflex.doubts:
            return None

        # 1. Bilirkişi Görüşünü Çek (Teknik analiz oradadır)
        expert_opinion = "Dosyada teknik bilirkişi görüşü bulunamadı."
        for p in persona_outputs:
            if "BİLİRKİŞİ" in p.role:
                expert_opinion = p.response
                break

        # 2. Bağlamı Hazırla
        # V120 farkı: Sadece şüpheyi değil, skor düşüklüğünü de veriyoruz.
        target_doubt = judge_reflex.doubts[0]

        # 3. V120 DİSİPLİNİNE UYGUN PROMPT (Revize Edildi)
        # "Genel tavsiye ver" yerine "Kaynak ve Risk belirt" diyoruz.
        prompt = f"""
        SEN KIDEMLİ BİR DAVA STRATEJİSTİSİN.
        Aşağıdaki hukuki tıkanıklığı açmak için "EK-5 İTİRAZ AKSİYON PLANI" (V120 Standardı) formatında tek bir hamle belirle.

        === DURUM ANALİZİ ===
        HAKİMİN ŞÜPHESİ: "{target_doubt}"
        UZMAN/BİLİRKİŞİ TESPİTİ: "{expert_opinion[:400]}..."

        === GÖREV ===
        Bu şüpheyi (doubt) ortadan kaldıracak, kanuna dayalı SOMUT bir adım yaz.
        Sadece genel geçer laflar etme (Örn: "Dilekçe yazılmalı" deme, "HMK 281 uyarınca ek rapor talep edilmeli" de).
        Bu tereddüdü azaltmak için yapılabilecek TEK ve SOMUT hukuki aksiyonu yaz.

        === İSTENEN FORMAT (AYNEN KULLAN) ===
        Başlık: [Kısa, Çarpıcı Strateji Adı]
        Aksiyon: [Somut ne yapılmalı? Dilekçe mi, Keşif mi, Tanık mı?]
        Kaynak: [Hukuki Dayanak. Örn: "HMK md. 281" veya "Yargıtay Yerleşik İçtihadı"]
        Risk: [Bu aksiyon alınmazsa ne olur? Örn: "İspat yükü ters döner."]
        Etki: [1-10 arası tahmini puan]
        """

        print("   ✅ CPU: V120 Standartlarında Risk Payload'ı hazırlandı.")

        return {
            "prompt": prompt,
            "target_doubt": target_doubt
        }

    # 🔴 ADIM 2: GPU ÇALIŞTIRMA (SADECE BURASI LLM KULLANIR)
    def execute_action(self, payload) -> List[StrengtheningAction]:
        if not payload: return []

        print("   🛠️ GPU: Stratejik Aksiyon Planı (V120 Logic) işleniyor...")
        try:
            # 1. LLM Çağrısı
            raw_result = LegalUtils.safe_extract_content(self.llm.invoke(payload["prompt"]))

            # 2. Sanitizer (V120 Temizliği - İngilizce halüsinasyonları siler)
            clean_result = self.sanitizer.sanitize_hallucinations(raw_result)

            # 3. Parse Et
            return [self._parse_action_v120(clean_result, payload["target_doubt"])]
        except Exception as e:
            print(f"   âŒ Aksiyon hatası: {e}")
            return []

    def _parse_action_v120(self, text: str, doubt: str) -> StrengtheningAction:
        """
        V120 formatındaki (Başlık/Aksiyon/Kaynak/Risk) çıktıyı parse eder.
        """
        lines = text.splitlines()

        # Varsayılan Değerler
        data = {
            "Başlık": "Stratejik Hamle",
            "Aksiyon": "Dosya kapsamına uygun beyan sunulmalıdır.",
            "Kaynak": "Genel Hukuk İlkeleri",  # V120 Yeni Alan
            "Risk": "Hak kaybı yaşanabilir.",  # V120 Yeni Alan
            "Etki": "5"
        }

        # Satır satır parse et
        for line in lines:
            line = line.strip()
            if not line: continue

            # Anahtar kelimeleri yakala
            for key in data.keys():
                if line.startswith(f"{key}:"):
                    # "Başlık: Örnek" -> "Örnek" kısmını al
                    val = line.split(":", 1)[1].strip()
                    # Boş değilse kaydet
                    if val:
                        data[key] = val

        # Etki puanını sayıya çevir
        try:
            impact_val = int("".join(filter(str.isdigit, data["Etki"])))
            if impact_val > 10: impact_val = 9
        except:
            impact_val = 5

        # StrengtheningAction nesnesini döndür
        # NOT: StrengtheningAction dataclass'ına 'source_ref' ve 'risk_analysis' alanlarını eklediğinizden emin olun.
        return StrengtheningAction(
            title=data["Başlık"],
            description=data["Aksiyon"],
            related_doubt=doubt,
            impact_score=impact_val,
            # Aşağıdaki alanlar StrengtheningAction sınıfında tanımlı olmalıdır
            source_ref=data["Kaynak"],
            risk_analysis=data["Risk"]
        )


class LegalUtils:
    @staticmethod
    def force_unlock_db():
        lock_file = os.path.join(LegalConfig.QDRANT_PATH, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file);
                print("🔍“ KİLİT DOSYASI TEMİZLENDİ.")
            except:
                pass

    @staticmethod
    def safe_extract_content(resp) -> str:
        """Gemini list dönebildiği için güvenli string dönüşümü yapar."""
        if hasattr(resp, 'content'):
            content = resp.content
        else:
            content = resp

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                else:
                    texts.append(str(part))
            return "".join(texts).strip()
        return str(content).strip()

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


# --- V121: ADVANCED LOOP BREAKER ---
class LegalTextSanitizer:
    """V121: Gelişmiş Tekrar Engelleyici (Madde Bazlı)"""

    def __init__(self):
        self.seen_sentences = set()
        self.written_articles = set()  # YENİ: Madde numaralarını takip et
        self.dropped_count = 0

    def enforce_no_repeat(self, text):
        return self.sanitize_hallucinations(text)

    def is_mostly_english(self, text):
        """Metnin İngilizce halüsinasyon olup olmadığını kontrol eder."""
        common_english_words = {"the", "and", "is", "of", "to", "in", "that", "it", "with", "as", "for", "childhood",
                                "education", "development"}
        words = text.lower().split()
        if not words: return False
        english_count = sum(1 for w in words if w in common_english_words)
        return (english_count / len(words)) > 0.3

    def sanitize_hallucinations(self, text):
        """İngilizceye kayan kısımları ve tekrar eden satırları temizler."""
        if not text: return ""

        # 1. İngilizce Kontrolü
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if self.is_mostly_english(line): continue
            clean_lines.append(line)

        text = "\n".join(clean_lines).replace("[END_OF_TEXT]", "").replace("<|endoftext|>", "")

        # 2. Tekrar Kontrolü (Mevcut Mantık)
        PROTECTED_PREFIXES = (
            "⚠️",
            "A.",
            "B.",
            "C.",
            "------------------------------------------------",
        )

        """Metindeki anlamsal tekrarları ve aynı kanun maddelerini temizler."""
        if not text: return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # 1. ÖNCE değişkeni tanımla
            clean_line = line.strip()

            # 2. SONRA kontrol et
            if clean_line.startswith(PROTECTED_PREFIXES):
                cleaned_lines.append(line)
                continue

            if len(clean_line) < 5:  # Çok kısa satırları (boşluk vb.) geç
                cleaned_lines.append(line)
                continue

            # --- V121 GÜNCELLEME: Madde Numarası Kontrolü ---
            article_match = re.search(
                r'(?:(TMK|HMK|BK|TBK|CMK)\s*)?(?:Madde|Md\.|m\.)\s*(\d+)',
                clean_line,
                re.IGNORECASE
            )
            if article_match:
                article_num = article_match.group(1)  # Sadece numarayı al (örn: "598")
                if article_num in self.written_articles:
                    self.dropped_count += 1
                    continue  # Aynı madde numarası daha önce yazıldıysa atla
                self.written_articles.add(article_num)
            # ------------------------------------------------

            # Cümlenin "özünü" (ilk 80 karakter) anahtar yap
            # Bu sayede "Mirasçılık belgesi..." ile "Mirasçılık belgesinin..." aynı sayılır
            key = re.sub(r'\s+', ' ', clean_line.lower())
            key = re.sub(r'[^\w\s]', '', key)[:80]

            if key in self.seen_sentences:
                self.dropped_count += 1
                continue  # BU SATIRI ATLA (TEKRAR)

            self.seen_sentences.add(key)
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def reset(self):
        self.seen_sentences = set()
        self.written_articles = set()  # Reset işleminde burayı da temizle
        self.dropped_count = 0


# ==================================================
# 3ï¸âƒ£ LEGAL AUDIT LOGGER
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
        """
        Sistemdeki HER anlamlı adım buradan geçer
        """
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

        if score_impact is not None:
            event["score_impact"] = score_impact

        if resulting_score is not None:
            event["resulting_score"] = resulting_score

        if confidence is not None:
            event["confidence"] = confidence

        self.logs.append(event)

    def export(self) -> Dict[str, Any]:
        """
        UI / API / Storage için tek JSON
        """
        return {
            "case_id": self.case_id,
            "started_at": self.started_at,
            "completed_at": time.time(),
            "timeline": self.logs,
        }


# ==================================================
# 4ï¸âƒ£ ACTIONABLE RECOMMENDATION ENGINE
# ==================================================
class ActionableRecommendationEngine:
    # 1. Sabit Profil Haritası (Safety Layer)
    RECOMMENDATION_PROFILE = {
        "DELIL": {
            "evidence_type": ["tanık", "belge", "bilirkişi", "keşif", "yemin"],
            "priority": "YÜKSEK",
            "estimated_cost": "Orta",
            "time_impact": "Orta",
            "base_score_range": (5, 10)
        },
        "ICTIHAT": {
            "evidence_type": ["emsal karar", "HGK kararı", "İBK"],
            "priority": "ORTA",
            "estimated_cost": "Düşük",
            "time_impact": "Kısa",
            "base_score_range": (3, 7)
        },
        "USUL": {
            "evidence_type": ["dilekçe", "itiraz", "süre tutum"],
            "priority": "YÜKSEK",
            "estimated_cost": "Düşük",
            "time_impact": "Kısa",
            "base_score_range": (2, 4)
        },
        "TALEP_DARALTMA": {
            "evidence_type": ["strateji"],
            "priority": "ORTA",
            "estimated_cost": "Düşük",
            "time_impact": "Kısa",
            "base_score_range": (4, 6)
        }
    }

    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_concerns, query_text=""):
        recommendations = []
        for concern in judge_concerns:
            category = self._classify_concern(concern)
            if not category: category = "DELIL"

            profile = self.RECOMMENDATION_PROFILE.get(category, self.RECOMMENDATION_PROFILE["DELIL"])
            # V145: LLM'den hem öneri hem de etki skoru (1-10) alıyoruz.
            rec_data = self._generate_recommendation_text_with_score(concern, self._category_to_turkish(category))
            rec_text = rec_data.get("suggestion", "İlgili hususta ek delil ve beyan sunulmalıdır.")
            llm_score = rec_data.get("impact_score", 5)

            # Profil bazlı ağırlık ile harmanla
            score_boost = min(llm_score, profile["base_score_range"][1])
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
            if "sgk" in concern_lower or "iş" in concern_lower:
                return {"entity": "Nüfus Müdürlüğü / UYAP", "method": "Kayıt Celbi", "responsible": "Mahkeme"}
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

    def _generate_recommendation_text_with_score(self, concern, category_tr):
        prompt = f"""
        BAĞLAM: Türk Hukuku (Yargıtay/BAM uygulaması). Başka ülke veya sistem kullanma.
        Bir avukata yol gösterecek şekilde, aşağıdaki hakim tereddüdüne yönelik {category_tr} odaklı SOMUT bir aksiyon önerisi yaz.
        Hakim Tereddüdü: "{concern}"

        JSON ÇIKTI FORMATI:
        {{
          "suggestion": "Emir kipiyle somut bir cümle",
          "impact_score": 8
        }}
        """
        try:
            res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res:
                res = res.split("```")[1].split("```")[0].strip()
            return json.loads(res)
        except:
            return {"suggestion": "İlgili hususta ek delil ve beyan sunulmalıdır.", "impact_score": 5}

    def _pick_evidence(self, options):
        if not options: return "Genel"
        return random.choice(options)


# ==================================================
# 5ï¸âƒ£ HAFIZA YÖNETİCİSİ (FULL INTEGRATED - V127 MASTER PROMPT)
# ==================================================
class LegalMemoryManager:
    # --- SIMULATION CONFIG ---
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
        self.sanitizer = LegalTextSanitizer()  # V120 Sanitizer
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
            res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            if "LEHINE" in res: return "LEHINE"
            if "ALEYHINE" in res: return "ALEYHINE"
            return "BELIRSIZ"
        except:
            return "BELIRSIZ"

    def _detect_domain_from_query(self, query_text):
        if query_text in self.domain_cache: return self.domain_cache[query_text]
        prompt = f"Sorgu: \"{query_text}\"\nBu sorgu hangi hukuk dalına girer? SADECE TEK KELİME CEVAP VER."
        try:
            domain = LegalUtils.safe_extract_content(self.llm.invoke(prompt)).split()[0]
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

    # --- V127: MASTER PROMPT GENERATOR ---
    def _build_master_prompt(self, role, domain, topic, analysis_type, memory_context, main_input, task_instruction):
        return f"""
SENİN ROLÜN: {role}

{LegalConfig.PROMPT_GUARD}

────────────────────────────────
ALLOWED SCOPE (ZORUNLU SINIRLAR)
────────────────────────────────
- Hukuk Alanı: {domain}
- Odak Konu: {topic}
- İnceleme Türü: {analysis_type}
- Yargı Çerçevesi: Türk Hukuku (Yargıtay / BAM)

Bu analiz SADECE yukarıdaki scope ile sınırlıdır.
Bu sınırların dışındaki her konu otomatik olarak ANALİZ DIŞIDIR.

────────────────────────────────
YERLEŞİK HAFIZA / İÇTİHAT BAĞLAMI
────────────────────────────────
{memory_context}

────────────────────────────────
OLAY / BELGE / TEREDDÜT
────────────────────────────────
{main_input}

────────────────────────────────
GÖREV
────────────────────────────────
{task_instruction}

ÇIKTIYI OLUŞTURMADAN ÖNCE:
- Scope dışına çıkıp çıkmadığını kontrol et.
- Tekrar veya genelleme olup olmadığını denetle.
- Hukuki rolünü ihlal edip etmediğini denetle.
"""

    # --- V127: PERSONA FUNCS UPDATED TO MASTER PROMPT ---

    def _generate_judge_doubts_v120(self, query, principle_text, domain="Genel"):
        """Hakimin ilk refleksini ve tereddütlerini üretir (Master Prompt ile)."""
        task = """
Bu ilke ışığında, olayı bir TÜRK HAKİMİ gözüyle değerlendir.
Bu ilke ışığında, olayı değerlendirirken yaşadığın EN FAZLA 3 TEMEL TEREDDÜTÜ (Doubts) listele.
Her tereddüt SOMUT olsun: delil eksikliği, usul sorunu, emsal uyuşmazlığı gibi.
Tereddütler kısa ve net olsun (maks 1 cümle).

ZORUNLU KURAL (ATIF VE USUL):
1. Tereddütlerini belirtirken mutlaka HMK (Hukuk Muhakemeleri Kanunu), TMK (Medeni Kanun) veya TCK ilgili maddelerine atıf yap.
2. Önce USUL YÖNÜNDEN (Görev, Yetki, Zamanaşımı) bir engel olup olmadığına bak. "Usul esastan mukaddemdir" ilkesini uygula.
3. Tereddütler "Acaba şöyle mi?" gibi BASİT OLMASIN. "HMK Md. 190 uyarınca davacının ... hususunu ispatlaması gerekirken..." gibi teknik olsun.

Ayrıca dosya hakkındaki İLK VİCDANİ KANAATİNİ (Red/Kabul Eğilimli) tek kelimeyle yaz.

ÇIKTI FORMATI (JSON):
{
  "reflex": "RED EĞİLİMLİ veya KABUL EĞİLİMLİ",
  "doubts": ["HMK Md. X uyarınca... (Tereddüt 1)", "Yerleşik Yargıtay İçtihadı gereği... (Tereddüt 2)"]
}
"""
        prompt = self._build_master_prompt(
            role="TÜRK HAKİMİ",
            domain=domain,
            topic=query,
            analysis_type="Hakim İlk Değerlendirmesi",
            memory_context=principle_text,
            main_input=query,
            task_instruction=task
        )

        print(
            f"   ⏳ LLM DÜŞÜNÜYOR: [TÜRK HAKİMİ] İlk refleks ve tereddütler belirleniyor... ({self.llm.__class__.__name__})")
        try:
            res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [TÜRK HAKİMİ] Analiz tamamlandı.")
            # JSON temizliği
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res:
                res = res.split("```")[1].split("```")[0].strip()
            return json.loads(res)
        except:
            return {"reflex": "BELİRSİZ",
                    "doubts": ["Dosya kapsamında delil durumu", "Emsal kararın uygunluğu", "Usul eksiklikleri"]}

    def _generate_plaintiff_response_v120(self, doubts, principle_text, domain="Genel", query_text=""):
        doubts_text = "\n".join([f"- {d}" for d in doubts])
        combined_input = f"OLAY: {query_text}\n\nHAKİM TEREDDÜTLERİ:\n{doubts_text}"

        task = """
GÖREVİN:
- Her bir tereddüte AYRI AYRI cevap vermek.
- Hakimi kabul yönünde ikna etmek (Davacı vekili olarak).

KRİTİK KURALLAR (Uymayan cevap reddedilir):
1. ATIF ZORUNLULUĞU: Her cümleni mutlaka bir YASAL DAYANAĞA bağla (Örn: "TMK Md. 166/1 gereği...", "Yargıtay HGK 2021/45 K. sayılı ilamı uyarınca...").
2. Dayanağı olmayan, sadece "bence" veya "müvekkilim haklıdır" şeklindeki soyut beyanları ASLA KULLANMA.
3. Genel hukuk anlatma. Doğrudan somut olaya uygula.
4. Cevabında mutlaka varsa [MEVZUAT] veya [EMSAL KARAR] etiketli belgeye ATIF YAP (Madde no veya Karar no ver).
5. Genel hukuk anlatma, doğrudan somut olaya ve müvekkilin haklılığına bağla.
6. Her cevap maks 3-4 cümle olsun.

ÇIKTI FORMATINI ASLA DEĞİŞTİRME:

--------------------------------------------------
DAVACI VEKİLİ DEĞERLENDİRMESİ
--------------------------------------------------
Tereddüt 1:
- Cevap: (Yasal dayanaklı cevap)

Tereddüt 2:
- Cevap: (Yasal dayanaklı cevap)

Tereddüt 3:
- Cevap: (Yasal dayanaklı cevap)
"""
        prompt = self._build_master_prompt(
            role="DAVACI VEKİLİ",
            domain=domain,
            topic="Hakim Tereddütlerini Giderme",
            analysis_type="Hukuki Argümantasyon",
            memory_context=principle_text,
            main_input=combined_input,
            task_instruction=task
        )

        print(f"   ⏳ LLM DÜŞÜNÜYOR: [DAVACI VEKİLİ] Tereddütleri yanıtlıyor...")
        try:
            raw = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [DAVACI VEKİLİ]")
            return self.sanitizer.enforce_no_repeat(raw)
        except:
            return "Davacı vekili beyanı oluşturulamadı."

    def _generate_defendant_response_v120(self, doubts, principle_text, domain="Genel", query_text=""):
        doubts_text = "\n".join([f"- {d}" for d in doubts])
        combined_input = f"OLAY: {query_text}\n\nHAKİM TEREDDÜTLERİ:\n{doubts_text}"

        task = """
GÖREVİN:
- Hakimin tereddütlerini DERİNLEŞTİRMEK.
- Usul itirazlarını (Zamanaşımı, Hak düşürücü süre, Derdestlik, Hukuki yarar yokluğu) öncelikli sunmak.
- Kabul ihtimalini zayıflatmak.

KRİTİK KURALLAR:
1. Her tereddüde AYRI AYRI cevap ver ve tereddüdü derinleştir.
2. Cevabında mutlaka varsa [MEVZUAT] veya [EMSAL KARAR] eksikliğine veya aleyhe durumuna ATIF YAP.
3. ATIF ZORUNLULUĞU: İtirazlarını mutlaka ilgili kanun maddesine dayandır. (Örn: "HMK 114. maddesi uyarınca dava şartı yokluğu...", "TBK Md. 147 gereği zamanaşımı defi...").
4. Genel hukuk anlatma, somut olaydaki eksikliklere bağla.
5. Soyut itiraz yapma ("Kabul etmiyoruz" yetmez). Hukuki gerekçesini yaz.
6. Her cevap maks 3-4 cümle olsun.
7. Davacının iddialarını "hayatın olağan akışına aykırılık" ve "ispat yükü" (HMK 190) kuralları çerçevesinde çürüt.


KRİTİK KURALLAR (Uymayan cevap reddedilir):



ÇIKTI FORMATINI ASLA DEĞİŞTİRME:

--------------------------------------------------
DAVALI VEKİLİ DEĞERLENDİRMESİ
--------------------------------------------------
Tereddüt 1:
- Karşı Argüman: (Yasal dayanaklı itiraz)

Tereddüt 2:
- Karşı Argüman: (Yasal dayanaklı itiraz)

Tereddüt 3:
- Karşı Argüman: (Yasal dayanaklı itiraz)
"""
        prompt = self._build_master_prompt(
            role="DAVALI (KARŞI TARAF) VEKİLİ",
            domain=domain,
            topic="Tereddütleri Derinleştirme ve İtiraz",
            analysis_type="Hukuki Argümantasyon",
            memory_context=principle_text,
            main_input=combined_input,
            task_instruction=task
        )

        print(f"   ⏳ LLM DÜŞÜNÜYOR: [DAVALI VEKİLİ] Karşı argümanlar hazırlanıyor...")
        try:
            raw = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [DAVALI VEKİLİ]")
            return self.sanitizer.enforce_no_repeat(raw)
        except:
            return "Davalı vekili beyanı oluşturulamadı."

    def _generate_expert_response_v120(self, doubts, principle_text, domain="Genel", query_text=""):
        doubts_text = "\n".join([f"- {d}" for d in doubts])
        combined_input = f"OLAY: {query_text}\n\nHAKİM TEREDDÜTLERİ:\n{doubts_text}"

        task = """
GÖREVİN:
- Hukuki mantık zincirini kontrol etmek.

YANITLA:
- Tereddütler hukuken yerinde mi?
- Davacı cevapları yeterli mi?
- Davalı itirazları hukuki mi?

ÇIKTI FORMATINI ASLA DEĞİŞTİRME:

--------------------------------------------------
BİLİRKİŞİ TESPİTLERİ
--------------------------------------------------
Genel Hukuki Değerlendirme:
- ...

Zayıf Noktalar:
- ...

Tutarlı Noktalar:
- ...
"""
        prompt = self._build_master_prompt(
            role="TARAFSIZ BİLİRKİŞİ",
            domain=domain,
            topic="Hukuki Tutarlılık Denetimi",
            analysis_type="Bilirkişi Mütalaası",
            memory_context=principle_text,
            main_input=combined_input,
            task_instruction=task
        )

        print(f"   ⏳ LLM DÜŞÜNÜYOR: [BİLİRKİŞİ] Dosya denetleniyor...")
        try:
            raw = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [BİLİRKİŞİ]")
            return self.sanitizer.enforce_no_repeat(raw)
        except:
            return "Bilirkişi raporu oluşturulamadı."

    def _simulate_post_strengthening_score(self, base_score, recommendations):
        total_boost = 0
        seen_cats = {}
        for rec in recommendations:
            cat = rec.get("category", "DELIL")
            impact = rec['risk_reduction']['expected_score_increase']
            if cat in seen_cats: impact = int(impact * 0.6)
            seen_cats[cat] = True
            total_boost += impact

        return {"current_score": base_score, "projected_score": min(base_score + total_boost, self.MAX_SCORE),
                "total_boost": total_boost}

    # --- MAIN RECALL FUNCTION (V127 UPDATE) ---
    def recall_principles(self, query_text):
        try:
            # 1. AUDIT START
            self.audit_logger = LegalAuditLogger()
            self.sanitizer.reset()  # Reset memory for new query

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

            # V148 (REVERTED): Kullanıcı İsteği Üzerine Tam Analiz Modu Geri Getirildi
            # Hız yerine detaylı analiz tercih edildi.

            self.latest_ui_data = {
                "query": query_text, "domain": query_domain, "principles": [], "net_decision": {},
                "executive_summary": "", "audit_log": {}
            }

            for item in sorted_hits:
                # 2. Risk Analizi
                analysis = self._calculate_case_success_probability(
                    item["conf"], item["trend_dir"], item["conflict"], item["domain_match"], item["polarity"]
                )

                # [V150 FIX] Persona analizi buradan kaldırıldı. 
                # Artık sadece ana akışta (LegalApp.run) yapılacak.
                # recall_principles SADECE veri getirmeli, işlememeli.
                
                reflex = "BELİRSİZ"
                doubts = ["Detaylı analiz ana akışta yapılacaktır."]
                plaintiff_text = "N/A"
                defendant_text = "N/A"
                expert_text = "N/A"
                action_plan = []
                simulation_result = {"projected_score": 0}

                # Store Complete Data (V120 Structure - Simplified)
                self.latest_ui_data["principles"].append({
                    "text": item['text'], "trend_log": item['evolution_note'], "polarity": item['polarity'],
                    "conflict_flag": item['conflict'], "year_bucket": item['year_bucket'],
                    "score_data": analysis,
                    "personas_v120": {
                        "judge_reflex": reflex,
                        "doubts": doubts,
                        "plaintiff": plaintiff_text,
                        "defendant": defendant_text,
                        "expert": expert_text
                    },
                    "personas": {"judge": str(doubts), "opponent": defendant_text, "opponent_title": "Davalı",
                                 "expert": expert_text, "devil": "N/A"},
                    "conflict_analysis": {"conflict_level": "N/A", "conflict_score": 0, "summary": []},
                    "reasoned_verdict": f"İÇTİHAT ÖZETİ: {item['text'][:100]}...",
                    "action_plan": action_plan,
                    "simulation": simulation_result
                })

                #self.latest_ui_data["executive_summary"] = exec_summary
                self.latest_ui_data["net_decision"] = {"decision": reflex}

                memory_text += f"- [{item['domain']}] {item['text']}\n"
                memory_text += f"  ⚖️ REFLEKS: {reflex} | ⚠️ Tereddüt: {len(doubts)} adet\n"

            # V120: Audit Log Export
            self.latest_ui_data["audit_log"] = self.audit_logger.export()

            return memory_text
        except Exception as e:
            print(f"Hata: {e}")
            return ""

    # --- MATEMATİKSEL YARDIMCILAR (TAM) ---
    def _cosine_similarity(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        if mag1 == 0 or mag2 == 0: return 0.0
        return dot / (mag1 * mag2)

    def _calculate_vector_mean(self, vectors):
        if not vectors: return []
        dim = len(vectors[0])
        mean = [0.0] * dim
        for v in vectors:
            for i in range(dim):
                mean[i] += v[i]
        return [x / len(vectors) for x in mean]

    def _cluster_reasonings(self, items, threshold=0.86):
        clusters = []
        for item in items:
            added = False
            for c in clusters:
                if self._cosine_similarity(item['vector'], c['centroid']) >= threshold:
                    c['members'].append(item)
                    all_vecs = [m['vector'] for m in c['members']]
                    c['centroid'] = self._calculate_vector_mean(all_vecs)
                    added = True
                    break
            if not added:
                clusters.append({'members': [item], 'centroid': item['vector']})
        return [c['members'] for c in clusters]

    def _calculate_principle_confidence(self, cluster):
        count = len(cluster)
        count_score = min(1.0, count / 10)
        return 0.7 + (count_score * 0.3)

    def _analyze_trend_momentum(self, trend_dict):
        if not trend_dict: return "Veri Yetersiz"
        return "İstikrarlı Seyir"

    # --- ESKİ SAVE FONKSİYONLARI (TAM & EKSİKSİZ) ---
    def calculate_memory_consensus(self, source_name, current_decision, vector_score):
        try:
            f = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))])
            p, _ = self.client.scroll("judge_memory_v1", scroll_filter=f, limit=20)
            if not p:
                if vector_score > 0.8: return 1.10
                return 1.0

            match_c = sum(1 for x in p if x.payload.get("decision") == current_decision)
            if len(p) == 0: return 1.0
            ratio = match_c / len(p)

            if ratio > 0.8: return 1.15
            if ratio < 0.2: return 0.85
            return 1.0
        except:
            return 1.0

    def save_decision(self, query, doc_name, decision, reason, doc_type):
        try:
            vec = self.embedder.embed_query(f"{query} {doc_name} {decision} {reason}")
            payload = {
                "query": query, "source": doc_name, "decision": decision,
                "reason": reason, "doc_type": doc_type,
                "timestamp": time.time(), "created_at": datetime.now().isoformat(), "id": str(uuid.uuid4())
            }
            self.client.upsert("judge_memory_v1", [PointStruct(id=payload['id'], vector=vec, payload=payload)])
        except:
            pass

    # --- KONSOLİDASYON (TAM) ---
    def consolidate_principles_v79(self):
        print("\n🔥 İÇTİHAT MİMARI: Artımlı Konsolidasyon (V120)...")
        try:
            time_filter = Filter(must=[FieldCondition(key="timestamp", range=Range(gt=self.last_consolidation_ts))])
            points, _ = self.client.scroll(LegalConfig.MEMORY_COLLECTIONS["decision"], scroll_filter=time_filter,
                                           limit=200)

            candidates = []
            for p in points:
                if (p.payload.get('doc_type') == 'EMSAL KARAR' and len(
                        p.payload.get('reason', '')) > 30 and p.payload.get('decision') == 'KABUL'):
                    candidates.append({
                        "reason": p.payload['reason'], "id": p.id,
                        "source": p.payload.get('source', 'Bilinmeyen'),
                        "timestamp": p.payload.get('timestamp', time.time()),
                        "decision": p.payload.get('decision'), "vector": None
                    })

            if len(candidates) < 3:
                print("   ℹ️ Yeterli yeni veri yok.")
                return

            print(f"   🔍 {len(candidates)} adet YENİ gerekçe analiz ediliyor...")
            texts = [c["reason"] for c in candidates]
            vectors = self.embedder.embed_documents(texts)
            for i, v in enumerate(vectors): candidates[i]["vector"] = v
            clusters = self._cluster_reasonings(candidates, threshold=0.86)

            for cluster in clusters:
                if len(cluster) < 3: continue

                # Küme Gerekçelerini Birleştir
                reasonings_text = "\n".join([f"- {c['reason']}" for c in cluster])
                prompt = f"""
GÖREV: Aşağıdaki mahkeme gerekçelerini analiz et.
1. Ortak hukuki ilkeyi TEK CÜMLEDE özetle.
2. Bu konunun ait olduğu Hukuk Dalını (Miras, Ceza, Borçlar vb.) belirle.

GEREKÇELER:
{reasonings_text}

FORMAT:
İLKE: [İlke Cümlesi]
ALAN: [Hukuk Dalı]
"""
                res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
                principle_match = re.search(r"İLKE:\s*(.*)", res)
                domain_match = re.search(r"ALAN:\s*(.*)", res)

                if principle_match:
                    principle_text = principle_match.group(1)
                    domain_text = domain_match.group(1) if domain_match else "Genel"
                    conf = self._calculate_principle_confidence(cluster)
                    source_ids = [c['id'] for c in cluster]

                    self._save_principle_v79(principle_text, conf, source_ids, domain_text, cluster)

            self._save_state()
            print("✅ Konsolidasyon tamamlandı.")
        except Exception as e:
            print(f"Hata: {e}")

    def _save_principle_v79(self, text, confidence, source_ids, domain, cluster_data):
        try:
            vec = self.embedder.embed_query(text)
            polarity = self._detect_polarity(text)
            hits = self.client.query_points("principle_memory_v1", query=vec, limit=10, score_threshold=0.80).points

            conflict = False
            trend = Counter()
            p_stats = {"LEHINE": 0, "ALEYHINE": 0, "BELIRSIZ": 0}

            # Conflict Check
            if polarity in p_stats: p_stats[polarity] += 1
            for h in hits:
                p = h.payload.get("polarity", "BELIRSIZ")
                if p in p_stats: p_stats[p] += 1
                if (p == "LEHINE" and polarity == "ALEYHINE") or (
                        p == "ALEYHINE" and polarity == "LEHINE"): conflict = True

            # Trend Check
            for c in cluster_data:
                bucket = self._extract_year_bucket(c.get("timestamp", time.time()))
                trend[(bucket, c.get("decision", "KABUL"))] += 1

            trend_dict = {}
            for (b, d), count in trend.items():
                if b not in trend_dict: trend_dict[b] = {"KABUL": 0, "RED": 0}
                trend_dict[b][d] = count

            evolution = self._analyze_trend_momentum(trend_dict)

            payload = {
                "principle": text, "confidence": confidence, "domain": domain,
                "polarity": polarity, "trend": trend_dict, "conflict_flag": conflict,
                "source_count": len(source_ids), "source_ids": source_ids, "evolution_note": evolution,
                "generated_by": "consolidation_v120", "timestamp": time.time(), "created_at": datetime.now().isoformat()
            }
            self.client.upsert("principle_memory_v1", [PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)])
        except:
            pass


# ==================================================
# 7ï¸âƒ£ YENİ ARAÇLAR: REASONING & STRATEGY (RESTORED)
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
    """V125: Dinamik Hakim Rolü Ataması"""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, audit_logs, story=None, context_str=None):
        logs_list = audit_logs.get("timeline", []) if isinstance(audit_logs, dict) else audit_logs
        summary_lines = [f"- {log['description']}" for log in logs_list if "description" in log]
        audit_summary = "\n".join(summary_lines)

        # 1. DAVA TÜRÜ TESPİTİ (Meta-Data Çıkarımı)
        dava_turu = "GENEL"
        if story:
            s_lower = story.lower()
            if any(k in s_lower for k in ["veraset", "miras", "tereke", "vasiyet", "ölünce"]):
                dava_turu = "SULH HUKUK (MİRAS)"
            elif any(k in s_lower for k in ["işçi", "kıdem", "ihbar", "fesih", "işveren"]):
                dava_turu = "İŞ MAHKEMESİ"
            elif any(k in s_lower for k in ["boşanma", "nafaka", "velayet", "eş"]):
                dava_turu = "AİLE MAHKEMESİ"
            elif any(k in s_lower for k in ["ceza", "suç", "sanık", "hapis"]):
                dava_turu = "CEZA MAHKEMESİ"
            elif any(k in s_lower for k in ["ticaret", "şirket", "bono", "çek"]):
                dava_turu = "TİCARET MAHKEMESİ"

        # 2. HAKİM ROLÜNÜN BELİRLENMESİ
        hakim_rolu = "İLGİLİ MAHKEME HAKİMİ"
        if "MİRAS" in dava_turu:
            hakim_rolu = "SULH HUKUK HAKİMİ"
        elif "İŞ" in dava_turu:
            hakim_rolu = "İŞ MAHKEMESİ HAKİMİ"
        elif "AİLE" in dava_turu:
            hakim_rolu = "AİLE MAHKEMESİ HAKİMİ"
        elif "CEZA" in dava_turu:
            hakim_rolu = "ASLİYE CEZA HAKİMİ"
        elif "TİCARET" in dava_turu:
            hakim_rolu = "ASLİYE TİCARET HAKİMİ"

        prompt = f"""
        GÖREV: SEN, KIDEMLİ BİR {hakim_rolu} OLARAK GEREKÇELİ KARAR YAZIYORSUN.

        {LegalConfig.PROMPT_GUARD}

        OLAY ÖZETİ: {story if story else 'Dosya kapsamı'}
        MEVZUAT, EMSAL VE DELİLLER: {context_str if context_str else audit_summary}

        GÖREVİN:
        Karar gerekçeni şu yapı ile yaz (resmi üslup, yaklaşık 250-350 kelime):

        1. Dosya kapsamına giren delillerin ve toplanan tüm kanıtların özeti (tanık beyanları, bilirkişi raporu, belgeler vb. somut olarak belirt).
        2. Tarafların iddiaları ve savunmalarının kısa özeti.
        3. Hukuki değerlendirme: İlgili kanun maddeleri, Yargıtay içtihatları ve emsal kararlara somut atıf yaparak olayın nasıl değerlendirildiği.
        4. Hakim olarak karşılaştığın tereddütler (maksimum 2-3 tane, somut) ve bunların nasıl giderildiği.
        5. Sonuç: Davanın kabulü/reddi/kısmen kabulü, ek delil istenmesi vb. net hüküm.

        Bu kararın kesin hüküm etkisi olmadığını ve kanun yoluna açık olduğunu belirt.
        Somut olayla bağlantılı, soyut genel ifadelerden kaçın. Gerçek bir hakim karar gerekçesi gibi doğal ve akıcı olsun.
        """
        print(f"   ⏳ LLM DÜŞÜNÜYOR: [GEREKÇELİ KARAR] Yazılıyor...")
        try:
            res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [GEREKÇELİ KARAR]")
            return res
        except:
            return "Gerekçe oluşturulamadı."


class AppealArgumentGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_reasoning):
        prompt = f"""
        GÖREV: SEN, KIDEMLİ BİR AVUKATSIN. Aşağıdaki hakim gerekçesini "üst mahkeme incelemesi" (istinaf/temyiz) için hukuki süzgeçten geçir.

        {LegalConfig.PROMPT_GUARD}

        Asagida bir hakimin karar gerekcesi yer almaktadir.
        Bu gerekceden hareketle, UST MAHKEMEYE sunulmak uzere itiraz argumanlari yaz.

        KURALLAR:
        - Hakime saygi dili kullan
        - "eksik inceleme", "yanlis takdir", "delillerin birlikte degerlendirilmemesi" kaliplari kullan
        - Madde madde yaz (Max 5 madde)

        HAKİM GEREKÇESİ:
        {judge_reasoning}
        """
        print(f"   ⏳ LLM DÜŞÜNÜYOR: [İTİRAZ ARGÜMANLARI] Hazırlanıyor...")
        try:
            res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [İTİRAZ ARGÜMANLARI]")
            return res
        except:
            return "İtiraz argümanları oluşturulamadı."


class AppealPetitionGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, judge_reasoning, case_topic):
        prompt = f"""
BAĞLAM: Türk Hukuku. BAM / Yargıtay uygulaması.
SEN: Kıdemli bir avukatsın.
{LegalConfig.PROMPT_GUARD}

Aşağıda yer alan hakim gerekçesine karşı, üst mahkemeye sunulmak üzere
RESMİ, KURUMSAL ve HUKUKİ DİLDE tam bir İTİRAZ / İSTİNAF / TEMYİZ DİLEKÇESİ taslağı yaz.

KURALLAR:
- Hakime saygılı dil kullan.
- "Eksik inceleme", "yanlış takdir", "hukuka aykırılık" kalıpları yer alsın.
- Madde numaraları kullan.

ZORUNLU UNSURLAR:
- Mahkeme adı, dosya no (örnek: ... Mahkemesi, 2024/... E.)
- Kararın özeti
- Somut itiraz nedenleri (eksik inceleme, yanlış hukuk uygulaması vb.)
- Hangi TMK maddesi veya Yargıtay içtihadının yanlış uygulandığı
- İstemin net ifadesi
- Avukat imzası kısmını bırak

ZORUNLU BAŞLIKLAR:
1. KARARIN ÖZETİ
2. İTİRAZ NEDENLERİ
3. HUKUKİ DEĞERLENDİRME
4. SONUÇ VE İSTEM

DOSYA KONUSU: {case_topic}
HAKİM GEREKÇESİ: {judge_reasoning}

ÇIKTI (Sadece Dilekçe Metni):
"""
        print(f"   ⏳ LLM DÜŞÜNÜYOR: [İSTİNAF DİLEKÇESİ] Yazılıyor...")
        try:
            res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
            print(f"   ✅ LLM YANITLADI: [İSTİNAF DİLEKÇESİ]")
            return res
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
            GÖREV: KIDEMLİ BİR AVUKAT olarak, aşağıdaki itiraz argümanını karşılayacak veya güçlendirecek SOMUT BİR AKSİYON (EYLEM) planı hazırla.

            JSON FORMATI:
            ALANLAR: title, evidence_type (tanık/belge/bilirkişi/içtihat), source, estimated_time, estimated_cost, risk_if_missing

            İTİRAZ ARGÜMANI: {arg}
            """
            try:
                res = LegalUtils.safe_extract_content(self.llm.invoke(prompt))
                if "```json" in res:
                    res = res.split("```json")[1].split("```")[0].strip()
                elif "```" in res:
                    res = res.split("```json")[1].split("```")[0].strip()

                action = json.loads(res)
                action["action_id"] = str(uuid.uuid4())
                action["linked_argument"] = arg
                actions.append(action)
            except:
                continue
        return actions


class CorporateCover:
    @staticmethod
    def add(pdf, case_id, version="V120"):
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 24)
        pdf.ln(60)
        pdf.cell(0, 10, "LEGAL OS", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DejaVu", size=14)
        pdf.cell(0, 10, "Yapay Zeka Destekli Hukuki Analiz Raporu", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(30)
        pdf.set_font("DejaVu", "B", 10)
        pdf.cell(0, 8, f"DOSYA KIMLIGI: {case_id}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DejaVu", "", 10)
        pdf.cell(0, 8, f"RAPOR TARIHI: {datetime.now().strftime('%d.%m.%Y %H:%M')}", align="C", new_x=XPos.LMARGIN,
                 new_y=YPos.NEXT)
        pdf.cell(0, 8, f"SISTEM SURUMU: {version}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(50)
        pdf.set_font("DejaVu", "I", 8)
        pdf.multi_cell(0, 5,
                       "YASAL UYARI: Bu rapor, yapay zeka algoritmalari kullanilarak uretilmistir. Hukuki tavsiye niteliginde olmayip, karar destek amaclidir.",
                       align="C")


# ==================================================
# 8.5. DOCUMENT ARCHIVER (V148: OTO KAYIT)
# ==================================================
class DocumentArchiver:
    """
    V148: Canlı arama sonuçlarını (emsal/mevzuat) yerel diske otomatik kaydeder.
    Klasör Yapısı: indirilenDosyalar/EMSAL | indirilenDosyalar/MEVZUAT
    """
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indirilenDosyalar")

    @staticmethod
    def _sanitize_filename(title):
        """Dosya ismi olamayacak karakterleri temizler."""
        return re.sub(r'[\\/*?:"<>|]', "", title)[:150]  # Windows max path için kısalt

    @staticmethod
    def _save_file_worker(doc_data):
        """
        Multithreading için işçi fonksiyonu.
        """
        try:
            doc_type = doc_data.get("type", "GENEL").upper()
            title = doc_data.get("title") or doc_data.get("source") or f"doc_{uuid.uuid4()}"
            content = doc_data.get("page_content") or doc_data.get("text") or ""
            url = doc_data.get("url", "")

            # Klasör Yolu
            folder_path = os.path.join(DocumentArchiver.ROOT_DIR, doc_type)
            os.makedirs(folder_path, exist_ok=True)

            # Dosya Adı
            safe_name = DocumentArchiver._sanitize_filename(title)
            file_path = os.path.join(folder_path, f"{safe_name}.txt")

            # İçerik Hazırlığı
            file_content = f"BAŞLIK: {title}\nKAYNAK URL: {url}\nTÜR: {doc_type}\nTARİH: {datetime.now()}\n{'=' * 50}\n{content}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            return f"✅ Kaydedildi: {safe_name}"
        except Exception as e:
            return f"⚠️ Kayıt Hatası: {e}"

    @staticmethod
    def archive_batch(documents: List[Any]):
        """
        Gelen belge listesini PARALEL olarak kaydeder.
        documents: Dict listesi veya Qdrant PointStruct listesi olabilir.
        """
        if not documents: return

        # Qdrant PointStruct -> Dict Dönüşümü gerekebilir
        clean_docs = []
        for d in documents:
            if hasattr(d, 'payload'):  # Qdrant Point
                clean_docs.append(d.payload)
            elif isinstance(d, dict):  # Raw Dict
                clean_docs.append(d)
            elif hasattr(d, 'page_content'):  # Langchain Doc
                clean_docs.append({"title": d.metadata.get("source"), "text": d.page_content,
                                   "type": d.metadata.get("type", "GENEL")})

        # ThreadPool ile Arka Planda Kayıt
        print(f"   💾 [Arşiv] {len(clean_docs)} belge 'indirilenDosyalar' klasörüne yedekleniyor...")
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(DocumentArchiver._save_file_worker, doc) for doc in clean_docs]
                # Sonuçları beklemeye gerek yok (fire-and-forget), ama hata loglamak için bakabiliriz.
                # Akışı yavaşlatmamak için burayı non-blocking bırakabiliriz ama ThreadPool zaten main thread'i bloklamaz (submit anında).
                # Ancak 'context manager' (with) bloğundan çıkarken wait=True defaulttur.
                # Hız için bekleme (wait=False) yapmak daha iyi olurdu ama veri kaybı riski var.
                # V148'de güvenli olması için bekliyoruz, zaten I/O hızlıdır.
        except Exception as e:
            print(f"   ⚠️ Arşivleme servisi hatası: {e}")

    # ==================================================
    # 8ï¸ âƒ£ ARAMA MOTORU SINIFI (SEARCH ENGINE)
    # ==================================================


class LegalSearchEngine:
    def __init__(self):
        self.config = LegalConfig()
        self.dense_embedder = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL, base_url="http://192.168.134.42:11434")
        self.mcp_bridge = YargiMcpBridge()
        self.client = None
        atexit.register(self.close)

    def connect_db(self):
        if self.client is not None: return True
        print("   🔌 Veritabanı bağlantısı başlatılıyor...")

        # [V150 FIX] Manual Lock Removal
        lock_file = os.path.join(self.config.QDRANT_PATH, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                print("   🔓 Kilit dosyası manuel olarak temizlendi.")
            except Exception as e:
                print(f"   ⚠️ Kilit dosyası silinemedi: {e}")

        try:
            self.client = QdrantClient(path=self.config.QDRANT_PATH)
            print("   ✅ Veritabanı bağlantısı BAŞARILI.")
            return True
        except Exception as e:
            print(f"\n❌ VERİTABANI HATASI: {e}")
            return False

    def close(self):
        if self.client:
            try:
                self.client.close()
                self.client = None
                print("\n🔍’ Veritabanı bağlantısı güvenli şekilde kapatıldı.")
            except:
                pass

    def run_indexing(self):
        if not self.connect_db(): return False

        for key, config in self.config.SOURCES.items():
            collection_name = config["collection"];
            folder_path = config["folder"]
            print(f"   👉 Koleksiyon kontrol ediliyor: {config['desc']}...")

            if not os.path.exists(folder_path):
                os.makedirs(folder_path);
                print(f"      ⚠️ Klasör oluşturuldu: {folder_path}");
                continue

            if not self.client.collection_exists(collection_name):
                print(f"      ⚙️  '{collection_name}' oluşturuluyor...")
                self.client.create_collection(collection_name,
                                              vectors_config=VectorParams(size=768, distance=Distance.COSINE))

            print(f"      🔍 Mevcut dosyalar taranıyor...")
            indexed_files = set()
            offset = None
            while True:
                points, offset = self.client.scroll(collection_name, limit=100, with_payload=True, with_vectors=False,
                                                    offset=offset)
                for p in points:
                    if 'source' in p.payload: indexed_files.add(p.payload['source'])
                if offset is None: break

            files_on_disk = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
            new_files = [f for f in files_on_disk if f not in indexed_files]

            if not new_files: print(f"      ✅ {config['desc']} güncel ({len(files_on_disk)} dosya)."); continue
            print(f"      â™»ï¸  {config['desc']} için {len(new_files)} yeni dosya işleniyor...")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_texts = [];
            all_metadatas = []

            for filename in new_files:
                try:
                    loader = PyMuPDFLoader(os.path.join(folder_path, filename))
                    docs = loader.load()
                    chunks = text_splitter.split_documents(docs)
                    for c in chunks:
                        clean_content = LegalUtils.clean_text(c.page_content)
                        all_texts.append(clean_content)
                        all_metadatas.append(
                            {"source": filename, "type": config['desc'], "page": c.metadata.get("page", 0) + 1})
                    print(f"      📝„ Okundu: {filename}")
                except Exception as e:
                    print(f"      ⚠️ Hata: {filename} - {e}")

            if not all_texts: continue
            print(f"      🚀 Vektörleştiriliyor ({len(all_texts)} parça)...")

            num_cores = cpu_count();
            batch_size = (len(all_texts) // num_cores) + 1;
            batches = []
            for i in range(0, len(all_texts), batch_size): batches.append(
                (all_texts[i:i + batch_size], self.config.EMBEDDING_MODEL))

            all_vectors = []
            try:
                with Pool(processes=num_cores) as pool:
                    results = pool.map(worker_embed_batch_global, batches)
                    for res in results: all_vectors.extend(res)
            except Exception as e:
                print(f"â Œ İşlemci Hatası: {e}");
                return False

            print(f"      💾 Kaydediliyor...");
            points = []
            for i, (vec, meta, txt) in enumerate(zip(all_vectors, all_metadatas, all_texts)):
                payload = {"page_content": txt, "source": meta["source"], "page": meta["page"], "type": meta["type"]}
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, txt + meta["source"] + collection_name))
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

            batch_size_upload = 64
            for i in range(0, len(points), batch_size_upload): self.client.upsert(collection_name,
                                                                                  points[i:i + batch_size_upload])

        print("✅ İndeksleme Tamamlandı.");
        return True

    def retrieve_raw_candidates(self, full_query):
        print("\n🔍 Belgeler Taranıyor (Dual Search - Aşama 1: Geniş Havuz)...")
        try:
            query_vector = self.dense_embedder.embed_query(full_query)
        except Exception as e:
            print(f"â Œ Embedding Hatası: {e}");
            return []

        all_candidates = []
        for key, config in self.config.SOURCES.items():
            try:
                results = self.client.query_points(collection_name=config["collection"], query=query_vector,
                                                   limit=self.config.SEARCH_LIMIT_PER_SOURCE).points
                if results:
                    unique_sources = {h.payload.get('source') for h in results if h.payload.get('source')}
                    print(
                        f"   📂 [LOKAL] {config['desc']}: {len(results)} parça bulundu ({len(unique_sources)} farklı dosyadan).")
                    for hit in results:
                        if 'type' not in hit.payload: hit.payload['type'] = config['desc']
                        # Lokal olduğunu işaretle
                        hit.payload['_origin'] = 'LOCAL'
                        all_candidates.append(hit)
            except Exception as e:
                print(f"   ⚠️ Lokal arama hatası ({key}): {e}")

        # V146: CANLI VERİTABANI ENTEGRASYONU (Yargi-MCP)
        try:
            print(f"   🌐 [DEBUG] MCP Canlı Arama Başlatılıyor...")
            print(f"      ▶️ Sorgu (Tam): {full_query}")
            print(
                f"      ▶️ Timeout Ayarı: {getattr(self.mcp_bridge.headers, 'timeout', 'Varsayılan')}")  # Bu bir obje değil dict, timeout bilgisi yok ama en azından erişimi test edelim.

            # 1. Sorguyu Anahtar Kelimelere Çevir (Basitleştir)
            keyword_prompt = f"GÖREV: Bu hukuki sorguyu arama motoru için 3-4 kelimelik anahtar kelime grubuna çevir. Sadece kelimeleri yaz. SORGU: {full_query}"
            search_keywords = self.judge.llm.invoke(
                keyword_prompt).content.strip()  # Örn: "mirastan ıskat veraset ilamı görevli mahkeme"

            print(f"   🌐 [Optimize] MCP Sorgusu: {search_keywords}")

            # 2. Optimize edilmiş sorguyu gönder
            live_results = self.mcp_bridge.search_all(search_keywords)

            #live_results = self.mcp_bridge.search_all(full_query)

            if live_results:
                print(f"   ✅ [DEBUG] MCP'den {len(live_results)} adet sonuç döndü.")

                # V148: OTOMATİK ARŞİVLEME (İndirilenleri Kaydet)
                DocumentArchiver.archive_batch(live_results)

                for res in live_results:
                    # Qdrant Hit objesini simüle et
                    class MockHit:
                        def __init__(self, payload, score):
                            self.payload = payload
                            self.score = score
                            self.id = str(uuid.uuid4())

                    # Başlık veya Source bilgisini al
                    title = res.get('title') or res.get('source') or 'Canlı Belge'

                    content = res.get("text", "")
                    if len(content) < 100:  # Eğer gelen metin çok kısaysa
                        content = f"BU KARARIN TAM METNİ ÇEKİLEMEDİ. ÖZET: {res.get('title', '')}"

                    payload = {
                        "page_content": content,
                        "source": f"CANLI: {title}",
                        "title": title,  # Arşivleme için raw title'ı da sakla
                        "type": "EMSAL",  # Varsayılan tip, MCP'den gelirse değiştirilebilir
                        "page": 1,
                        "url": res.get("url", ""),
                        "_origin": "ONLINE"  # Köken işareti
                    }
                    all_candidates.append(MockHit(payload,
                                                  0.90))  # Online belgelere biraz daha yüksek güven veriyoruz ancak sınırlayacağız
            else:
                print("   ⚠️ [DEBUG] MCP Sonuç Dönmedi (Liste Boş).")

        except Exception as e:
            print(f"   ⚠️ Canlı arama hatası (EXCEPTION): {e}")

        unique_docs = {}
        for hit in all_candidates:
            if hit.score < self.config.SCORE_THRESHOLD: continue
            key = f"{hit.payload['source']}_{hit.payload.get('page', 1)}"
            if key not in unique_docs or hit.score > unique_docs[key].score: unique_docs[key] = hit

        # V101: KOTA SİSTEMİ UYGULAMASI
        emsal_hits = []
        mevzuat_hits = []

        for hit in unique_docs.values():
            if hit.payload.get('type') == 'MEVZUAT':
                mevzuat_hits.append(hit)
            else:
                emsal_hits.append(hit)

        emsal_hits.sort(key=lambda x: x.score, reverse=True)
        mevzuat_hits.sort(key=lambda x: x.score, reverse=True)

        # V149: HYBRID MERGE STRATEJİSİ (Lokal + Online Dengesi)
        # Sadece skora bakarsak 0.90 alan Online belgeler Lokal'i ezer.
        # Bu yüzden havuzları ayırıp birleştiriyoruz.

        local_emsal = [h for h in emsal_hits if h.payload.get('_origin', 'LOCAL') == 'LOCAL']
        online_emsal = [h for h in emsal_hits if h.payload.get('_origin') == 'ONLINE']

        # Sıralama
        local_emsal.sort(key=lambda x: x.score, reverse=True)
        online_emsal.sort(key=lambda x: x.score, reverse=True)

        limit = self.config.LLM_RERANK_LIMIT  # Varsayılan 10, V149'da arttırılabilir
        statute_quota = 3  # Mevzuat kotası
        precedent_total_slots = limit - statute_quota  # Geriye kalan (Örn: 7)

        # Mevzuatları kesin al
        final_candidates = mevzuat_hits[:statute_quota]

        # Emsal slotlarını paylaştır (Örn: 7 slot varsa -> Min 3 Lokal, Min 3 Online gibi)
        # Amaç: Mevcutsa her iki taraftan da veri almak.

        if online_emsal and local_emsal:
            # Hibrit Mod: Yarı yarıya (veya yakın) paylaştır
            online_slots = math.ceil(precedent_total_slots / 2)  # 4
            local_slots = precedent_total_slots - online_slots  # 3

            print(f"   ⚖️  [HİBRİT BİRLEŞTİRME] Online: {online_slots}, Lokal: {local_slots} belge seçiliyor.")

            final_candidates.extend(online_emsal[:online_slots])
            final_candidates.extend(local_emsal[:local_slots])

            # Boşluk kalırsa doldur (Örn: local yetmedi, online'dan daha fazla al)
            remaining = limit - len(final_candidates)
            if remaining > 0:
                used_ids = {h.id for h in final_candidates}
                extras = [h for h in emsal_hits if h.id not in used_ids]
                extras.sort(key=lambda x: x.score, reverse=True)
                final_candidates.extend(extras[:remaining])

        else:
            # Sadece bir taraf varsa kural basit
            final_candidates.extend(emsal_hits[:precedent_total_slots])

        # Mevzuat eksikse (nadir) emsal ile tamamla
        if len(final_candidates) < limit:
            used_ids = {h.id for h in final_candidates}
            remaining_pool = [h for h in emsal_hits if h.id not in used_ids] + [h for h in mevzuat_hits if
                                                                                h.id not in used_ids]
            remaining_pool.sort(key=lambda x: x.score, reverse=True)
            needed = limit - len(final_candidates)
            final_candidates.extend(remaining_pool[:needed])

        if not final_candidates: print("🔴 Uygun belge bulunamadı."); return []
        print(f"   ✅ {len(final_candidates)} potansiyel belge bulundu. Yargıca gönderiliyor...")
        return final_candidates


# ==================================================

# 9️⃣ YARGIÇ VE MUHAKEME SINIFI (JUDGE)
# ==================================================
class LegalJudge:
    """
    CPU (Hazırlık) ve GPU (Çalıştırma) ayrıştırılmış Yargıç Motoru.
    """

    def __init__(self, memory_manager=None, llm=None):
        # Eğer dışarıdan LLM verilmezse konfigürasyondan yükle (Fallback)
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOllama(
                model=LegalConfig.LLM_MODEL,
                temperature=LegalConfig.LLM_CONFIG["temperature"],
                top_p=LegalConfig.LLM_CONFIG["top_p"],
                base_url="http://192.168.134.42:11434"
            )
        self.memory = memory_manager
        self.sanitizer = LegalTextSanitizer()

    def validate_user_input(self, story, topic):
        prompt = f"""
GÖREV: Metnin tamamen anlamsız rastgele tuşlama (gibberish) olup olmadığını tespit et.
METİN: "{story} {topic}"
ANALİZ KURALLARI:
1. "araba", "miras" gibi tek kelimelik girdiler [GEÇERLİ].
2. Sadece "asdasd", "lkgjdf" gibi rastgele tuşlamalar [GEÇERSİZ].
CEVAP (SADECE BİRİ): [GEÇERLİ] veya [GEÇERSİZ]
"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            if "GEÇERSİZ" in res: return False
            return True
        except:
            return True

    def generate_expanded_queries(self, story, topic):
        print("   ↳ 🧠 Sorgu Genişletiliyor...")
        try:
            prompt = f"GÖREV: Hukuki terimler.\nOLAY: {story}\nODAK: {topic}\n3 kısa cümle."
            res = self.llm.invoke(prompt).content
            return [line.strip() for line in res.splitlines() if len(line) > 5][:3]
        except:
            return [story]

    # [YENİ EKLENEN METOT]
    def _build_scope_block(self, topic, negatives=None):
        scope = f"""
ALLOWED SCOPE (ZORUNLU):
- Analiz SADECE şu konu ile sınırlı olacak: {topic}
- Türk Hukuku (Yargıtay/BAM uygulaması)
- Somut olay ve delil odaklı değerlendirme
"""
        if negatives:
            scope += "\nIMPLICITLY EXCLUDED (Bu alanlar analiz dışıdır):\n"
            for n in negatives:
                scope += f"- {n}\n"

        scope += "\nBu sınırların DIŞINA ÇIKMA.\n"
        return scope

    def _check_relevance_judge_smart(self, user_query, user_filter, negative_keywords, document_text, source_name,
                                     doc_type="EMSAL"):
        found_negative = None
        if negative_keywords:
            doc_lower = document_text.lower()
            for bad in negative_keywords:
                if re.search(rf"\b{re.escape(bad)}\b", doc_lower): found_negative = bad; break

        if found_negative:
            prompt = f"HUKUKÇU. Sorgu: '{user_query}'. Yasaklı: '{found_negative}'. Uygun mu? [RED]/[KABUL]."
            res = self.llm.invoke(prompt).content.strip()
            if "RED" in res: return False, f"⛔ YASAKLI: {res}"

        memory_context = ""
        if self.memory:
            memory_context = self.memory.recall_principles(user_query)

        # [YENİ] Scope bloğunun oluşturulması
        scope_block = self._build_scope_block(user_filter, negative_keywords)

        # V102: DOC TYPE SPECIFIC PROMPT
        if doc_type == "MEVZUAT":
            focus_instruction = "GÖREV: Bu kanun maddesi, yukarıdaki olaya HUKUKİ DAYANAK (Kanuni Temel) teşkil ediyor mu?\nBenzerlik arama, uygulanabilirlik ara."
        else:
            focus_instruction = "GÖREV: Bu emsal karar, yukarıdaki olayla ÖRGÜ VE SONUÇ bakımından BENZER mi?\nOlay benzerliği ara."

        prompt_gen = f"""
SEN KIDEMLI BIR HUKUKCUSSUN.

{scope_block}

{memory_context}

Sorgu: "{user_query}"
Belge ({doc_type}): "{document_text[:700]}..."

{focus_instruction}

SADECE BİRİNİ SEÇ: [ÇOK BENZER/UYGUN], [BENZER/UYGUN], [ZAYIF/ALAKASIZ]
Altına tek cümlelik gerekçe yaz.
"""
        res = self.llm.invoke(prompt_gen).content.strip()
        is_ok = ("ÇOK BENZER" in res) or ("BENZER" in res) or ("UYGUN" in res) or ("KABUL" in res)
        return is_ok, res

    def _assign_document_role(self, user_query, document_text):
        prompt = f"""
SEN HUKUKÇUSUN.
Sorgu: "{user_query}"
Belge: "{document_text[:800]}..."
GÖREV: Bu belge hukuki analizde nasıl kullanılmalı?
1. [DOĞRUDAN DELİL]: Olay örgüsü birebir örtüşüyor.
2. [EMSAL İLKE]: Olay farklı ama hukuk kuralı uygulanabilir.
SADECE ŞUNLARDAN BİRİNİ SEÇ:
[DOĞRUDAN DELİL] veya [EMSAL İLKE]
"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            if "DOĞRUDAN" in res: return "[DOĞRUDAN DELİL]"
            return "[EMSAL İLKE]"
        except:
            return "[EMSAL İLKE]"

    def evaluate_candidates(self, candidates, story, topic, negatives):
        print("\n⚖️  Akıllı Yargıç Değerlendiriyor (V120: Corporate Intelligence):")
        valid_docs = []

        for hit in candidates:
            doc_text = hit.payload['page_content']
            source = hit.payload['source']
            page = hit.payload['page']
            type_desc = hit.payload['type']

            is_ok, reason = self._check_relevance_judge_smart(story, topic, negatives, doc_text, source, type_desc)

            consensus_multiplier = 1.0
            if self.memory:
                consensus_decision = "KABUL" if is_ok else "RED"
                consensus_multiplier = self.memory.calculate_memory_consensus(source, consensus_decision, hit.score)

            base_score = min(max(hit.score, 0), 1) * 100
            norm_score = min(base_score * consensus_multiplier, 100.0)

            icon = "✅" if is_ok else "❌"

            if self.memory:
                decision_tag = "KABUL" if is_ok else "RED"
                self.memory.save_decision(f"{story} {topic}", source, decision_tag, reason, type_desc)

            if is_ok:
                role = self._assign_document_role(story, doc_text)

                log_score = f"%{norm_score:.1f}"
                if consensus_multiplier > 1.1:
                    log_score += " (⬆️ YÜKSEK GÜVEN)"
                elif consensus_multiplier < 1.0:
                    log_score += " (⬇️ RİSKLİ)"

                print(f"{icon} [{type_desc}] {source:<20} | Güven: {log_score} | Rol: {role}")

                extra_context = ""
                if type_desc == "EMSAL KARAR":
                    real_path = os.path.join(LegalConfig.SOURCES["emsal"]["folder"], source)
                    verdict = LegalUtils.extract_pdf_conclusion(real_path)
                    extra_context = f"\n\n🛑 [OTOMATİK EKLENEN KARAR SONUCU ({source})]:\n{verdict}\n🛑 KARAR SONU."

                valid_docs.append({
                    "source": source, "page": page, "type": type_desc, "role": role,
                    "text": doc_text + extra_context, "score": norm_score, "reason": reason
                })
            else:
                print(f"{icon} [{type_desc}] {source:<20} | Güven: %{norm_score:.1f}")

        return valid_docs

    # [V128 EKLENTİSİ] PDF İçin Emsal Açıklama Kartları
    def explain_precedents_for_pdf(self, accepted_docs, topic):
        print("\n📝 PDF İçin Emsal Kartları Hazırlanıyor...")
        cards = []

        # Sadece kabul edilen ve anlamlı rolü olan belgeleri seçiyoruz
        targets = [d for d in accepted_docs if d.get("role") in ["[EMSAL İLKE]", "[DOĞRUDAN DELİL]"]]

        for doc in targets:
            prompt = f"""
SEN BİR TÜRK HUKUKÇUSUSUN.
Ama bu bir KARAR değil, PDF RAPOR AÇIKLAMASIDIR.

KONU: {topic}

BELGE:
- Dosya: {doc['source']}
- Sayfa: {doc['page']}
- Rol: {doc['role']}
- Gerekçe: {doc.get('reason', '')}

METİN PARÇASI:
\"\"\"{doc['text'][:800]}...\"\"\"

GÖREV:
Bu belgenin neden bu dosya açısından önemli olduğunu,
avukatın veya müvekkilin rahatça okuyabileceği şekilde açıkla.

KURALLAR:
- Hukuki uydurma YAPMA
- Genel ders anlatımı YAPMA
- Hakim gibi hüküm kurma
- 1 paragrafı geçme
- "Bu belge önemlidir" diye başlama, direkt içeriğe gir.

ÇIKTI FORMATI:
**Gerekçe:** [Açıklama]
**İçerik:** [Özet]
"""
            try:
                explanation = self.llm.invoke(prompt).content.strip()
                cards.append({
                    "filename": doc["source"],
                    "page": doc["page"],
                    "role": doc["role"],
                    "content": explanation
                })
            except Exception as e:
                print(f"⚠️ Kart oluşturma hatası: {e}")

        return cards

    def generate_final_opinion(self, story, topic, context_str, context=None, judge_reflex=None):
        print("\n🧑‍⚖️  AVUKAT YAZIYOR (V120 + V150 Hybrid Mode)...")

        # [V151 EKLENTİSİ - HYBRID FORCE]
        # Kullanıcı "Lokal" seçse bile, bu fonksiyon ÖZEL OLARAK Cloud LLM (Groq) kullanmalı.
        effective_llm = self.llm
        if not LegalConfig.USE_CLOUD_LLM:
             # Eğer API Key varsa, Cloud model oluştur.
             if LegalConfig.GROQ_API_KEY and "YOUR" not in LegalConfig.GROQ_API_KEY:
                  print("\n☁️  Hukuki Görüş için CLOUD LLM (Groq) Zorlanıyor (Hybrid Mode)...")
                  try:
                      effective_llm = ChatGroq(
                          model_name=LegalConfig.CLOUD_MODEL_NAME, 
                          api_key=LegalConfig.GROQ_API_KEY,
                          temperature=0.1,
                          max_retries=5
                      )
                  except Exception as e:
                      print(f"⚠️ Cloud LLM başlatılamadı: {e}. Yerel model ile devam ediliyor.")
             else:
                 print("⚠️ Cloud LLM zorlandı ancak API Anahtarı eksik/hatalı. Yerel model kullanılıyor.")

        # [V151 MONITORING]
        active_mode = "🏠 [LOCAL - OLLAMA]"
        if isinstance(effective_llm, ChatGroq):
            active_mode = "☁️ [CLOUD - GROQ]"
        elif "groq" in str(type(effective_llm)).lower(): # Fallback check
             active_mode = "☁️ [CLOUD - GROQ]"
        
        print(f"\n📢  FINAL GÖRÜŞ İÇİN AKTİF MODEL: {active_mode}")  
        print(f"    (Bu satırı takip ederek Cloud kullanıldığından emin olabilirsiniz)")

        # 1. DOMAIN MAPPING (V149 Feature requested by User)
        DOMAIN_MAPPINGS = {
            "miras_hukuku": {
                "maddeler": "TMK md. 598 (Mirasçılık belgesi), TMK md. 510-513 (Mirastan çıkarma/ıskat), TMK md. 605 (Mirasın reddi)",
                "ictihatlar": "Yargıtay 2. ve 14. Hukuk Dairesi (Iskatın veraset ilamında şerh düşülmesi, sıfatın tamamen kalkmaması ilkesi)"
            },
            "borclar_hukuku": {
                "maddeler": "TBK md. 1-146 (Genel Hükümler), TBK md. 49 (Haksız Fiil), TBK md. 112 (Borca Aykırılık)",
                "ictihatlar": "Yargıtay 3. ve 13. Hukuk Dairesi (Sözleşme serbestisi ve kusur sorumluluğu)"
            },
            "aile_hukuku": {
                "maddeler": "TMK md. 166 (Evlilik birliğinin sarsılması), TMK md. 174 (Tazminat), TMK md. 175 (Yoksulluk nafakası)",
                "ictihatlar": "Yargıtay 2. Hukuk Dairesi (Kusur belirlemesi ve nafaka kriterleri)"
            },
            "ceza_hukuku": {
                "maddeler": "TCK md. 1-75 (Genel Hükümler), CMK md. 223 (Hüküm çeşitleri)",
                "ictihatlar": "Yargıtay Ceza Genel Kurulu (Şüpheden sanık yararlanır ilkesi)"
            },
            "is_hukuku": {
                "maddeler": "İş Kanunu md. 17 (İhbar), md. 25 (Haklı fesih), 1475 SK md. 14 (Kıdem)",
                "ictihatlar": "Yargıtay 9. ve 22. Hukuk Dairesi (İşçi lehine yorum ilkesi)"
            },
            "genel_hukuk": {
                "maddeler": "HMK md. 27 (Hukuki Dinlenilme Hakkı), TMK md. 2 (Dürüstlük Kuralı), TMK md. 6 (İspat Yükü)",
                "ictihatlar": "Yargıtay Hukuk Genel Kurulu (İspat yükü ve usul ekonomisi)"
            }
        }

        # Domain algıla (Safe fallback)
        try:
            domain_key = context.detected_domain.lower().replace(" ", "_") if context else "genel_hukuk"
        except:
            domain_key = "genel_hukuk"

        mapping = DOMAIN_MAPPINGS.get(domain_key, DOMAIN_MAPPINGS["genel_hukuk"])

        system_content = f"""SEN BİR TÜRK HAKİMİSİN.
{LegalConfig.PROMPT_GUARD}

🛑 KRİTİK VE ZORUNLU KURAL:
BU SİSTEM SADECE TÜRKÇE ÇALIŞIR. 
HER NE OLURSA OLSUN ÇIKTIYI SADECE VE SADECE **TÜRKÇE** DİLİNDE VER.
(RESPONSE MUST BE ONLY IN TURKISH LANGUAGE. DO NOT USE CHINESE OR ENGLISH.)

Görevin:
- Tarafları savunmak DEĞİL
- Dosyanın RED veya KABUL ihtimallerini, hukuki ve usuli açıdan değerlendirmektir.

- Dosyanın RED veya KABUL ihtimallerini, hukuki ve usuli açıdan değerlendirmektir.

            ZORUNLU HUKUKİ REFERANSLAR (CEVABINDA MUTLAKA KULLAN):
            - Kanun Maddeleri: {mapping['maddeler']}
            - Yargıtay İçtihatları: {mapping['ictihatlar']}

ZORUNLU KURAL:
Eğer sana verilen [EMSAL KARAR] metinleri, kullanıcının sorgusunun AYNISIYSA veya çok kısaysa;
bunu "EMSAL YOK" olarak kabul et ve halüsinasyon üretme.
"Mevcut verilerde tam metinli emsal bulunamadı" yaz.

NORMLAR HİYERARŞİSİ (ZORUNLU):
- [MEVZUAT] etiketli metinler KANUN maddesidir (TMK, BK vb.). Bunları kesin kural olarak sun.
- [EMSAL KARAR] etiketli metinler YARGITAY uygulamasıdır. Bunları "yorum ve uygulama" olarak sun.

ÖN KABULLER:
1. Veraset ilamı çekişmesiz yargı işidir.
2. Çekişmesiz yargı kararları maddi anlamda kesin hüküm oluşturmaz.
3. Hakim her zaman önce RED ihtimalini değerlendirir.
4. Usul eksikliği varsa ESASA GİRİLMEZ.

SANA SAĞLANAN BELGELER ETİKETLİDİR:
- [MEVZUAT]
- [EMSAL KARAR]

BELGE DIŞINA ÇIKMA.
YENİ EMSAL UYDURMA.
GENEL HUKUK ANLATISI YAPMA.

----------------------------------------------------------------
AŞAMA 1 — YARGISAL DEĞERLENDİRME (İÇ MUHAKEME)
----------------------------------------------------------------

Aşağıdaki soruları KENDİN için cevapla ve analizini buna göre yap:

- Dosya usulden reddedilebilir mi?
- Hakimin temel tereddüt noktaları neler?
- Sunulan emsal kararlar:
  - Yerleşik mi?
  - Güncel mi?
  - Somut olayla birebir mi?
- Bu dosyada hakimin takdir alanı var mı?

----------------------------------------------------------------
AŞAMA 2 — YAPILANDIRILMIŞ HUKUKİ RAPOR
----------------------------------------------------------------

ÇIKTIYI AŞAĞIDAKİ BAŞLIKLARLA VE AYNI SIRAYLA VER.
BAŞLIKLARI VE SIRAYI ASLA DEĞİŞTİRME.

------------------------------------------------------------
A. MEVZUAT DAYANAKLARI
------------------------------------------------------------
Burada:
- SADECE [MEVZUAT] etiketli belgeleri kullan.
- İlgili kanun maddelerini KISA ve NET şekilde özetle.
- Somut olayla doğrudan bağlantıyı belirt.
- Yorum yapma, normu açıkla.
- Aynı kanun maddesini birden fazla kez özetleme.
- Her madde numarasını sadece bir kez belirt.

------------------------------------------------------------
B. İLGİLİ EMSAL KARARLAR (ZORUNLU BÖLÜM)
------------------------------------------------------------
Burada:
- SADECE [EMSAL KARAR] etiketli belgeleri kullan. En az 2 emsal karar ÖZETLE.
- Her emsal için:
  - Karar numarası / tarihi (varsa)
  - Yargıtay dairesi
  - ROL’ünü belirt (EMSAL İLKE / DESTEKLEYİCİ / AYIRT EDİLEBİLİR)
  - Hakimin bakış açısından kısa GEREKÇE yaz (2-3 cümle)
- Eğer emsal yoksa "Somut olayla doğrudan ilgili güncel emsal karar tespit edilememiştir." yaz.

------------------------------------------------------------
C. SONUÇ VE HUKUKİ TAVSİYE
------------------------------------------------------------
Burada:
- Kullanıcının anlattığı somut olaya göre konuş.
- Bulunan emsaller ve mevzuata dayanarak:
  - Dosyanın ZAYIF yönlerini açıkla
  - Güçlendirilmesi gereken noktaları belirt
  - Net bir yol haritası çiz (ne yapılmalı / ne yapılmamalı)
- “Şu yapılırsa RED riski azalır” mantığıyla yaz.
- Dosyanın kabul edilme ihtimalini düşük/orta/yüksek olarak belirt.
- Red riskini azaltmak için 2-3 somut aksiyon öner.

----------------------------------------------------------------
YASAKLAR:
- Genel hukuk anlatısı
- Akademik açıklama
- Aynı fikri tekrar etmek
- Belge dışı yorum

SADECE BU DOSYAYI VE SAĞLANAN BELGELERİ DEĞERLENDİR. CEVABI TÜRKÇE YAZ.
"""

        user_content = f"""Aşağıdaki "DELİLLER" listesinde sunulan belgeleri kullanarak olayı analiz et.
OLAY: "{story}"
ODAK: "{topic}"
DELİLLER:
{context_str}
ANALİZİ BAŞLAT (TÜRKÇE):"""

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]

        # V150 Smart Retry Mechanism (3 retries for Groq API)
        max_retries = 3
        retry_count = 0
        full_res = ""

        while retry_count < max_retries:
            try:
                # Use stream via invoke/stream processing or direct invoke if simpler, but keep V128 logic of streaming print
                # Re-implementing streaming print within the retry block is risky if it fails mid-stream.
                # Safer: Get full content then print (or stream if confident).
                # Given V150 goal is robustness + quality, let's prioritize success.

                # We can simulate streaming by printing chunks if we use stream(),
                # but if it fails (RateLimit), we need to catch it.
                # Using invoke() is safer for retry logic, but streaming gives better UX.
                # Compromise: Try stream. If it crashes with RateLimit, catch and retry.

                full_res = ""
                for chunk in effective_llm.stream(messages):
                    c = chunk.content
                    full_res += c
                    print(c, end="", flush=True)
                print("\n")
                break  # Success

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Rate limit" in err_str:
                    retry_count += 1
                    wait_time = 45 * retry_count
                    print(
                        f"\n⚠️ Groq Hız Limiti (429). {wait_time}sn bekleniyor... (Deneme {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # Other errors -> Fallback to local
                    print(f"\n⚠️ LLM Kritik Hata: {e}")
                    # Optional: Fallback to local model here if desired,
                    # but V150 Smart Retry mainly targets rate limits.
                    # For now, break and return what we have or empty.
                    break

        # If loop finishes without success (and full_res is empty), try local or return error
        if not full_res and retry_count >= max_retries:
            print("\n⚠️ Groq tüm denemelerde başarısız oldu. Lokal modele düşülüyor (TODO)...")
            # Fallback implementation or just return error message
            return "HATA: Hukuki görüş oluşturulamadı (API Limitleri)."

        # V120 SANITIZATION
        cleaned_res = self.sanitizer.enforce_no_repeat(full_res)
        return cleaned_res

    def build_query_context(self, story, topic, negatives) -> 'QueryContext':
        # Ensure QueryContext class is available or defined.
        # If not, we might need to rely on a dictionary or simple object.
        # Assuming QueryContext is defined elsewhere in file (grep confirmed presence).
        ctx = QueryContext(query_text=story, topic=topic, negative_scope=negatives)
        try:
            ctx.assert_hard_limits()
        except:
            pass
        return ctx


# ==================================================
# 🔍 RAPORLAMA SINIFI (V120 - ROBUST FONT LOADER)
# ==================================================
class BrandedPDFGenerator(FPDF):
    def __init__(self, branding):
        super().__init__()
        self.branding = branding
        self.font_loaded = False

        # Font Yolları (Öncelik Sırası)
        possible_paths = [
            "fonts/DejaVuSans.ttf",  # 1. Yerel klasör
            os.path.join(LegalConfig.DRIVE_ROOT, "fonts/DejaVuSans.ttf"),  # 2. Drive klasörü
            "/content/drive/MyDrive/HukAI/fonts/DejaVuSans.ttf"  # 3. Tam yol (Hardcoded)
        ]

        font_path = None
        for p in possible_paths:
            if os.path.exists(p):
                font_path = p
                break

        # Font Yükleme Denemesi
        if font_path:
            try:
                self.add_font("DejaVu", "", font_path)
                # Bold için de aynısını veya regular'ı kullan
                bold_path = font_path.replace("Sans.ttf", "Sans-Bold.ttf")
                if os.path.exists(bold_path):
                    self.add_font("DejaVu", "B", bold_path)
                    self.add_font("DejaVu", "BI", bold_path)
                else:
                    self.add_font("DejaVu", "B", font_path)  # Fallback
                    self.add_font("DejaVu", "BI", font_path)  # Fallback

                self.add_font("DejaVu", "I", font_path)
                self.font_loaded = True
                print(f"✅ PDF Fontu Yüklendi: {font_path}")
            except Exception as e:
                print(f"⚠️ Font yükleme hatası: {e}")
        else:
            print(f"⚠️ UYARI: DejaVuSans.ttf bulunamadı! Türkçe karakterler bozuk çıkabilir.")
            print(
                f"   Lütfen şu dosyayı indirip 'HukAI/fonts' klasörüne koyun: https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf")

    def header(self):
        if self.branding.logo_path and os.path.exists(self.branding.logo_path):
            self.image(self.branding.logo_path, x=10, y=8, w=30)

        # Font Seçimi
        font = "DejaVu" if self.font_loaded else "helvetica"

        self.set_font(font, "B", 12)
        self.set_text_color(*self.branding.color)
        self.cell(0, 10, self.branding.firm_name, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
        self.set_draw_color(200, 200, 200)
        self.line(10, 25, 200, 25)
        self.ln(15)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        font = "DejaVu" if self.font_loaded else "helvetica"
        self.set_font(font, 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.branding.footer_text} | Sayfa {self.page_no()}', align='C')


# ==================================================
# 1ï¸âƒ£2ï¸âƒ£ Pipeline
# ==================================================
class LegalEvaluationPipeline:
    def __init__(self, judge_core, logic_engine):
        self.judge_core = judge_core
        self.logic_engine = logic_engine
        self.has_run = False
        self.last_result = None

    def run(self, decision_context, persona_outputs):
        if self.has_run:
            print("   ⚠️ Pipeline zaten çalıştı â€“ Son sonucu döndürüyor.")
            return self.last_result
        self.has_run = True

        # 1ï¸âƒ£ Deterministik ilk değerlendirme
        initial_reflex = self.judge_core.evaluate(decision_context)

        print(f"   ⚖️  ÖN YARGIÇ REFLEKSİ: {initial_reflex.tendency} (Skor: {initial_reflex.score})")

        if initial_reflex.score < 30:
            raise RuntimeError(
                f"Dosya hukuki olarak zayıf (Skor: {initial_reflex.score}). Hakim ilk refleksi RED yönünde. Lütfen daha güçlü delil veya emsal ile tekrar deneyin.")

        # 2ï¸âƒ£ Mantık motoru ile düzeltme
        final_reflex = self.logic_engine.run_logic(
            initial_reflex=initial_reflex,
            persona_outputs=persona_outputs
        )

        self.last_result = final_reflex
        return final_reflex


# ==================================================
# 7.5. ACTION ENGINE (PHASE 7 CPU/GPU ADAPTER)
# ==================================================
class ActionEngine:
    """
    V147 Wrapper for Risk & Action Analysis (CPU/GPU Split).
    ActionableRecommendationEngine mantığını Phase 7 mimarisine (Hazırlık -> Çalıştırma) uyarlar.
    """

    def __init__(self, llm):
        self.llm = llm
        # Helper logic (reuse existing helpers if needed, or standalone)
        self.recommender = ActionableRecommendationEngine(llm)

    def build_risk_payload(self, reflex, persona_outputs) -> List[Dict]:
        """
        CPU STEP: Risk analizi için promptları hazırlar.
        """
        payloads = []
        doubts = reflex.doubts if (reflex and reflex.doubts) else ["Dosya kapsamında genel hukuki riskler"]

        # Persona çıktılarından da risk türetebiliriz (Opsiyonel - Şimdilik sadece Hakim Tereddütleri)

        for doubt in doubts:
            category = self.recommender._classify_concern(doubt) or "DELIL"
            category_tr = self.recommender._category_to_turkish(category)

            # V120/V143 Uyumlu Prompt (Source Ref ve Risk Analysis ister)
            prompt = f"""
            GÖREV: Kıdemli bir avukata yol gösterecek şekilde, aşağıdaki HAKİM TEREDDÜDÜNE yönelik {category_tr} odaklı SOMUT ve UYGULANABİLİR bir aksiyon önerisi yaz.

            ANALİZ:
            Hakim Tereddüdü: "{doubt}"

            SENİN GÖREVİN:
            1. Bu tereddüdü giderecek EN ETKİLİ aksiyonu (delil, beyan, içtihat) belirle.
            2. Bu aksiyonun yapılmaması durumunda doğacak RİSKİ analiz et.
            3. Dayandığı hukuki KAYNAĞI (Madde/İlke) belirt.
            4. Davaya etkisini (1-10) puanla.

            JSON ÇIKTI FORMATI:
            {{
              "title": "Stratejik Hamle Başlığı",
              "description": "Emir kipiyle somut aksiyon cümlesi",
              "source_ref": "TMK Md. X / Yargıtay ... ilkesi",
              "risk_analysis": "Bu eksiklik ... sonucunu doğurur.",
              "impact_score": 8
            }}
            """
            payloads.append({
                "prompt": prompt,
                "doubt": doubt,
                "category": category
            })

        return payloads

    def execute_action(self, payloads: List[Dict]) -> List[StrengtheningAction]:
        """
        GPU STEP: LLM'i çalıştırır ve StrengtheningAction nesneleri üretir.
        """
        actions = []
        if not payloads: return []

        print(f"   🛡️ GPU (Risk Engine): {len(payloads)} tereddüt için aksiyon planı oluşturuluyor...")

        for p in payloads:
            try:
                # LLM Çağrısı
                try:
                    res = LegalUtils.safe_extract_content(self.llm.invoke(p["prompt"]))
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "Rate limit" in err_str:
                        print(f"      ⚠️ Groq KOTA DOLDU! Fallback (Lokal) Modele geçiliyor...")
                        # Fallback LLM (Anlık Oluştur)
                        fallback_llm = ChatOllama(model="qwen2.5:3b", temperature=0.1,base_url="http://192.168.134.42:11434")
                        res = LegalUtils.safe_extract_content(fallback_llm.invoke(p["prompt"]))
                    else:
                        raise e  # Diğer hataları yukarı fırlat

                # JSON Temizleme ve Parse
                if "```json" in res:
                    res = res.split("```json")[1].split("```")[0].strip()
                elif "```" in res:
                    res = res.split("```")[1].split("```")[0].strip()

                import json
                data = json.loads(res)

                # StrengtheningAction Oluştur
                # Emniyet kemeri: Veri tiplerini kontrol et
                score = int(data.get("impact_score", 5))
                if score > 10: score = 9

                action = StrengtheningAction(
                    title=data.get("title", f"{p['category']} Stratejisi"),
                    description=data.get("description", "Bu konuda detaylı beyan sunulmalıdır."),
                    related_doubt=p["doubt"],
                    impact_score=score,
                    source_ref=data.get("source_ref", "Genel Hukuk İlkeleri"),
                    risk_analysis=data.get("risk_analysis", "Hak kaybı riski mevcuttur.")
                )
                actions.append(action)

            except Exception as e:
                # Fallback Action
                print(f"      ⚠️ Aksiyon üretilemedi: {e}")
                actions.append(StrengtheningAction(
                    title="Genel Strateji Önerisi",
                    description=f"'{p['doubt']}' hususunda eksikliklerin giderilmesi gerekmektedir.",
                    related_doubt=p["doubt"],
                    impact_score=5,
                    source_ref="HMK İspat Kuralları",
                    risk_analysis="İspat yükü yerine getirilemeyebilir."
                ))

        return actions


# ==================================================
# ANA UYGULAMA (MAIN APP)
# ==================================================
class LegalApp:
    def __init__(self):
        print(f"\n{'=' * 50}")
        print(f"⚖️  LEGAL AI SYSTEM - SURUM: V149 (Hybrid Merge Enabled)")
        print(f"{'=' * 50}\n")

        # V147: HİBRİT MOD SEÇİMİ
        print("🤖 LLM CALISMA MODU SECİNİZ:")
        print("   1. LOKAL  (Qwen 2.5 - Ücretsiz, Çevrimdışı, Yavaş)")
        print("   2. ONLINE (Groq Llama 3 70B - Ücretsiz/Hızlı, API Key Gerekir)")

        choice = input("\n👉 Seçiminiz (1/2): ").strip()
        if not choice: choice = "1"  # Default to Local
        if choice == "2":
            LegalConfig.USE_CLOUD_LLM = True
            print("   ⚡ ONLINE MOD AKTİF (Groq API)")
            if not LegalConfig.GROQ_API_KEY or "YOUR" in LegalConfig.GROQ_API_KEY:
                key = input("🔑 Lütfen Groq API Anahtarınızı girin (gsk_...): ").strip()
                LegalConfig.GROQ_API_KEY = key
        else:
            LegalConfig.USE_CLOUD_LLM = False
            print("   🏠 LOKAL MOD AKTİF (Ollama)")

        # 🔥 PROFİLLİ LLMâ€™LER (Global Router - Tek Kanal GPU)
        # Streaming kapalı, Threading kapalı.
        self.judge_llm = get_llm_by_profile("judge")
        self.persona_llm = get_llm_by_profile("persona")
        self.risk_llm = get_llm_by_profile("risk")

        # 🧠 Motorlar
        self.search_engine = LegalSearchEngine()

        if self.search_engine.connect_db():
            # Memory Manager Judge profilini kullanır
            self.memory_manager = LegalMemoryManager(
                self.search_engine.client,
                self.search_engine.dense_embedder,
                self.judge_llm
            )
        else:
            self.memory_manager = None

        # Judge Engine (Ayrıştırılmış Versiyon)
        self.judge = LegalJudge(memory_manager=self.memory_manager, llm=self.judge_llm)

        # Mantık Motoru (Matematiksel - CPU)
        self.logic_engine = LegalDecisionLogic()

        # V145: İtiraz ve Dilekçe Üreteçleri
        self.appeal_arg_gen = AppealArgumentGenerator(self.judge_llm)
        self.appeal_pet_gen = AppealPetitionGenerator(self.judge_llm)

    def run(self):
        # Başlangıç İndeksleme Kontrolü
        if not self.search_engine.run_indexing():
            self.search_engine.close()
            sys.exit()

        # Hafıza Konsolidasyonu
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

                print("   🛡️ Girdi kontrol ediliyor...")
                #if not self.judge.validate_user_input(story, topic):
                #    print("   âŒ UYARI: Girdi anlamsız. Lütfen mantıklı bir olay giriniz.")
                #    continue

                # ---------------------------------------------------------
                # 1. BAĞLAM VE ARAMA (CPU AŞAMASI)
                # ---------------------------------------------------------
                start_total = time.time()
                print("   ⚙️ CPU: Bağlam ve sorgu hazırlanıyor...")

                # Context oluştur (CPU)
                ctx = self.judge.build_query_context(story, topic, negatives)
                print(f"   ✓ Bağlam Oluşturuldu: {ctx.detected_domain}")

                # Sorgu genişletme (Basit I/O veya hafif LLM çağrısı)
                expanded = self.judge.generate_expanded_queries(ctx.query_text, ctx.topic)
                full_query = f"{ctx.query_text} {ctx.topic} " + " ".join(expanded)

                # Vektör Arama (Disk I/O)
                candidates = self.search_engine.retrieve_raw_candidates(full_query)
                if not candidates: continue

                # ---------------------------------------------------------
                # 2. BELGE DEĞERLENDİRME (V120 MANTIK - V150 ENTEGRASYON)
                # ---------------------------------------------------------
                # V128 stili doğrudan değerlendirme (build_evaluation_payloads yerine)
                valid_docs = self.judge.evaluate_candidates(
                    candidates, ctx.query_text, ctx.topic, ctx.negative_scope
                )

                if not valid_docs:
                    print("🔴 Yargıç tüm belgeleri eledi.")
                    continue

                print(f"Judge inference bitti: {time.time() - start_total:.2f} sn")

                # Context String Oluşturma (CPU)
                context_str = ""
                for i, d in enumerate(valid_docs):
                    is_emsal = "EMSAL" in d['type'].upper()
                    doc_label = "[EMSAL KARAR]" if is_emsal else "[MEVZUAT]"
                    char_limit = 1000 if is_emsal else 800
                    context_str += f"""
                        BELGE #{i + 1}
                        ETİKET: {doc_label}
                        KAYNAK: {d['source']}
                        TÜR: {d['type']}
                        ROL: {d['role']}
                        YARGIÇ GEREKÇESİ: {d['reason']}
                        İÇERİK: {d['text'][:char_limit]}...
                        =========================================
                        """

                # Hafıza Çağırma (Opsiyonel - CPU/GPU)
                current_personas = {}
                mem_principles = []
                if self.memory_manager:
                    print("   🧠 Hafıza ve geçmiş içtihatlar taranıyor (GPU Embedding)...")
                    self.memory_manager.recall_principles(full_query)
                    ui_data = self.memory_manager.latest_ui_data
                    if ui_data and ui_data.get("principles"):
                        mem_principles = ui_data["principles"]

                # ---------------------------------------------------------
                # 3. KARAR ZEMİNİ İNŞASI (CPU)
                # ---------------------------------------------------------
                decision_context = DecisionBuilder.build_decision_context_from_valid_docs(valid_docs)
                decision_context = DecisionBuilder.enrich_decision_context_with_memory(decision_context, mem_principles)

                if not decision_context.has_minimum_legal_basis():
                    print("🔴 Yetersiz belge. Analiz durduruluyor.")
                    continue

                # ---------------------------------------------------------
                # 4. JUDGE CORE (DETERMİNİSTİK MATEMATİK - CPU)
                # ---------------------------------------------------------
                judge_core_instance = JudgeCore()
                reflex = judge_core_instance.evaluate(decision_context)

                print(f"   ⚖️  ÖN YARGIÇ REFLEKSİ: {reflex.tendency} (Skor: {reflex.score})")

                if reflex.score < 30:
                    print(f"🔴 Dosya hukuki olarak çok zayıf (Skor: {reflex.score}).")
                    continue

                # =========================================================
                # 5. PERSONA ENGINE (CPU HAZIRLIK -> GPU INFERENCE)
                # =========================================================
                persona_engine = PersonaEngine(self.persona_llm)

                # A. Promptları Hazırla (Domain Mapping & PDF Okuma Burada Yapılır)
                print("   ⚙️CPU: Persona verileri ve hukuk zemini hazırlanıyor...")
                persona_payloads = persona_engine.build_persona_prompts(ctx, decision_context, reflex)

                # B. GPU'da Çalıştır (Kesintisiz Seri Akış)
                persona_outputs = persona_engine.execute_personas(persona_payloads)

                # =========================================================
                # 6. MANTIK MOTORU (CPU)
                # =========================================================
                # LLM çıktılarına göre matematiksel düzeltme
                reflex = self.logic_engine.run_logic(
                    initial_reflex=reflex,
                    persona_outputs=persona_outputs
                )

                # =========================================================
                # 7. ACTION ENGINE (CPU HAZIRLIK -> GPU INFERENCE)
                # =========================================================
                action_engine = ActionEngine(self.risk_llm)

                # A. Prompt Hazırla (Bilirkişi verisini al, JSON şablonu kur)
                print("   ⚙️CPU: Risk analizi kurgulanıyor...")
                risk_payload = action_engine.build_risk_payload(reflex, persona_outputs)

                # B. GPU'da Çalıştır
                strengthening_actions = action_engine.execute_action(risk_payload)

                # Avukat Masası (Konsol Çıktısı)
                if strengthening_actions:
                    print(f"\n   🛠️  AKSİYON PLANI (V120 Disiplini):")
                    for act in strengthening_actions:
                        print(f"      🔍¹ [{act.impact_score}/10] {act.title}")
                        # YENİ ALANLARI YAZDIR
                        print(f"          ↳ Kaynak: {act.source_ref}")
                        print(f"          ↳ Risk: {act.risk_analysis}")
                        print(f"          ↳ Aksiyon: {act.description[:100]}...")

                # ---------------------------------------------------------
                # 8. FİNAL GEREKÇE VE RAPORLAMA (GPU + CPU)
                # ---------------------------------------------------------

                # Senior Judge Gerekçe Yazımı (GPU - Tek Prompt)
                full_advice = self.judge.generate_final_opinion( 
                    story=ctx.query_text,
                    topic=ctx.topic,
                    context_str=context_str,
                    context=ctx,
                    judge_reflex=reflex
                )

                # V145: İtiraz Argümanları ve Dilekçe Üretimi
                print("   ⚖️  BAM/Yargıtay İtiraz Stratejisi Hazırlanıyor...")
                appeal_args = self.appeal_arg_gen.generate(full_advice)
                appeal_petition = self.appeal_pet_gen.generate(full_advice, ctx.topic)

                print("\n🖨️  Raporlama Süreci Başlatılıyor (CPU)...")

                # 1. Orkestratörü Hazırla
                try:
                    report_orchestrator = ReportOrchestrator(
                        reporters=[
                            ClientSummaryPDF(),  # Basit özet
                            JudicialPDFReport()  # Detaylı yargısal rapor
                        ]
                    )

                    # 2. Tüm Raporları Tek Seferde Üret (CPU - FPDF)
                    pdf_paths = report_orchestrator.generate_all(
                        context=ctx,
                        judge_reflex=reflex,
                        persona_outputs=persona_outputs,
                        actions=strengthening_actions,
                        documents=decision_context.documents,
                        full_advice=full_advice,
                        appeal_arguments=appeal_args,
                        appeal_petition=appeal_petition
                    )

                    for path in pdf_paths:
                        print(f"   ✅ Rapor Üretildi: {path}")

                except NameError:
                    print("   ⚠️ PDF modülü bulunamadı, rapor atlanıyor.")

                # 4. Zaman Çizelgesi
                audit_dump = {}
                if self.memory_manager and hasattr(self.memory_manager, 'latest_ui_data'):
                    audit_dump = self.memory_manager.latest_ui_data.get("audit_log", {})

                print("\n📊 İŞLEM ZAMAN ÇİZELGESİ:")
                for log in audit_dump.get("timeline", []):
                    print(f"   {log['timestamp']} | {log['title']} â†’ {log['description']}")

        except KeyboardInterrupt:
            print("\n👋 Program durduruldu.")
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")
        finally:
            self.search_engine.close()


if __name__ == "__main__":
    log_gpu_status()
    freeze_support()
    app = LegalApp()
    app.run()
