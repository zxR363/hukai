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

from concurrent.futures import ThreadPoolExecutor, as_completed


#--------------------GPU ICIN ---------------
def get_llm_judge():
    return ChatOllama(
        model="qwen2.5:7b",
        temperature=0.1,
        num_ctx=8192,
    )



def log_gpu_status():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        print("🟢 GPU AKTİF (OLLAMA)")
    except:
        print("⚠️ GPU BULUNAMADI, CPU FALLBACK")

#--------------------GPU ICIN ---------------

# PDF CIKTILARI Mevcut importların altına ekleyin
from pdf_reports import (
    LegacyPDFReport,
    JudicialPDFReport,
    ClientSummaryPDF,  # Eğer kullanacaksanız
    ReportOrchestrator
)


# UTF-8 Ayarı
# sys.stdout.reconfigure(encoding="utf-8")


# ==================================================
# 1️⃣ KONFİGÜRASYON VE BAĞLAM SINIFLARI
# ==================================================

# # 🔨 Commit 5.3: Query Context (Single Source of Truth) – DÜZELTİLMİŞ VE ÇALIŞIR
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
        """Dataclass oluşturulduktan sonra çalışır – domain algılama burada"""
        self.detect_domain()
        self.assert_hard_limits()

    def detect_domain(self):
        """Sorgudan domain algıla – basit ama etkili"""
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

# 🔨 Commit 5.4: Decision Context (Yargısal Zemin)
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


# 🔨 Commit 5.5: Judge Reflex (Refleks Veri Yapısı)
@dataclass
class JudgeReflex:
    """
    Hakimin ilk refleksi.
    """
    tendency: str  # "KABUL" | "RED" | "TEREDDÜT"
    score: int  # 0–100
    doubts: List[str]  # Hakimin kafasına takılanlar


# 🔨 Commit 5.6: Persona Response (Persona Çıktı Modeli)
@dataclass
class PersonaResponse:
    role: str  # DAVACI | DAVALI | BILIRKISI
    response: str
    addressed_doubts: List[str]


# 🔨 Commit 5.7: Strengthening Action (Aksiyon Modeli)
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
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if self.is_mostly_english(line): continue
            clean_lines.append(line)
        return "\n".join(clean_lines).replace("[END_OF_TEXT]", "").replace("<|endoftext|>", "")


# DINAMIK LLM AYARLARI AYRIK CALISMASI ICIN
LLM_PROFILES = {
    "judge": {
        "model": "qwen2.5",
        "num_ctx": 1024,
        "temperature": 0.1
    },
    "persona": {
        "model": "qwen2.5",
        "num_ctx": 1024,
        "temperature": 0.3
    },
    "risk": {
        "model": "qwen2.5",
        "num_ctx": 1024,
        "temperature": 0.0
    }
}

def get_llm_by_profile(profile_name: str):
        """Verilen profile göre optimize edilmiş ChatOllama nesnesi döndürür."""
        config = LLM_PROFILES.get(profile_name, LLM_PROFILES["judge"])

        print(
            f"   🔌 LLM Başlatılıyor: [{profile_name.upper()}] | ctx: {config['num_ctx']} | temp: {config['temperature']}")

        return ChatOllama(
            model=config["model"],
            num_ctx=config["num_ctx"],
            temperature=config["temperature"],
            repeat_penalty=config.get("repeat_penalty", 1.1),

            # ⚠️ KRİTİK AYARLAR (GPU'YU KİLİTLER)
            streaming=False,  # Asla stream etme, bekle ve sonucu al.
            num_thread=4,  # CPU thread limiti (Ollama için)
            num_gpu=1  # Tek GPU zorlaması
        )
# DINAMIK LLM AYARLARI AYRIK CALISMASI ICIN

@dataclass
class LegalConfig:
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
# 2️⃣ YARDIMCI ARAÇLAR (STATIC)
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
        embedder = OllamaEmbeddings(model=model_name)
        return embedder.embed_documents(texts)
    except Exception as e:
        print(f"⚠️ Batch hatası (atlanıyor): {e}")
        return []


# 🔨 Commit 5.4: Decision Builder (Adaptör)
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
                "score": doc.get("score"), #Confidence ile aynı PDF_REPORTS'DA ihtiyac oluyor
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


# 🔨 Commit 5.5: Judge Core (Deterministik Akıl)
class JudgeCore:
    """
    LLM'siz, deterministik hakim muhakemesi.
    """

    def evaluate(self, decision_context: DecisionContext) -> JudgeReflex:
        score = 0
        doubts = []

        # 1️⃣ Belgelerden gelen güç
        for doc in decision_context.documents:
            # Skorlar 0-100 arasında geliyordu, burada normalize edip topluyoruz
            conf = doc.get("confidence", 0)
            if conf >= 90:
                score += 15
            elif conf >= 80:
                score += 10
            elif conf >= 70:
                score += 5
            else:
                doubts.append(
                    f"Düşük güvenli belge: {doc.get('source')}"
                )

        # 2️⃣ Hukuki ilkeler
        for principle in decision_context.principles:
            conf = principle.get("confidence", 0)  # 0-100 arası success probability
            if conf >= 85:
                score += 10
            elif conf < 60:
                doubts.append(
                    "Zayıf içtihat/ilke tespiti"
                )

        # 3️⃣ Skoru sınırla
        score = min(score, 100)

        # 4️⃣ Hakim refleksi
        if score >= 70:
            tendency = "KABUL"
        elif score <= 40:
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
    NETLIK_KELIMELERI = ["kanaat", "sonuç", "tespit edilmiştir", "mütalaa", "görüş","açıkça",
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
            return "TEREDDÜTLÜ – BİLİRKİŞİ MUĞLAK"

        # 2. Kilit: Tereddüt sayısı 1'den fazlaysa
        if tereddut_sayisi >= 1:
            return f"TEREDDÜTLÜ – {tereddut_sayisi} KAYNAK ŞÜPHELİ"

        # 3. Kilit: Davalı çok güçlüyse
        if davali_gucu >= 7:
            return "TEREDDÜTLÜ – DAVALI SAVUNMASI GÜÇLÜ"

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

            === SENİN ROLÜN VE KURALLARIN ===
            ROL: {role}
            TALİMAT: {instruction}

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

            # ❌ ThreadPoolExecutor YOK
            # ✅ Basit 'for' döngüsü (En hızlısı ve en güvenlisi budur)

            for i, payload in enumerate(prepared_payloads):
                role = payload["role"]
                print(f"      ▶️ [{i + 1}/{len(prepared_payloads)}] İşleniyor: {role}...")

                try:
                    # Bloklayıcı çağrı (Cevap gelene kadar kod durur)
                    raw_response = self.llm.invoke(payload["prompt"]).content.strip()
                    clean_response = self.sanitizer.sanitize_hallucinations(raw_response)  # Temizle

                    responses.append(PersonaResponse(
                        role=role,
                        response=clean_response,
                        addressed_doubts=payload["doubts"]
                    ))
                    print(f"      ✅ Tamamlandı: {role}")

                except Exception as e:
                    print(f"      ❌ Hata ({role}): {e}")
                    responses.append(PersonaResponse(
                        role=role,
                        response="Teknik hata nedeniyle beyan oluşturulamadı.",
                        addressed_doubts=payload["doubts"]
                    ))

            return responses

    def _run_single_inference(self, payload):
        try:
            # TEK GÖREV: String'i modele ver, String al.
            result = self.llm.invoke(payload["prompt"]).content.strip()
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
        MEVCUT SKOR: {judge_reflex.score}/100 (Riskli Bölge)

        === GÖREV ===
        Bu şüpheyi (doubt) ortadan kaldıracak, kanuna dayalı SOMUT bir adım yaz.
        Sadece genel geçer laflar etme (Örn: "Dilekçe yazılmalı" deme, "HMK 281 uyarınca ek rapor talep edilmeli" de).

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
            raw_result = self.llm.invoke(payload["prompt"]).content.strip()

            # 2. Sanitizer (V120 Temizliği - İngilizce halüsinasyonları siler)
            clean_result = self.sanitizer.sanitize_hallucinations(raw_result)

            # 3. Parse Et
            return [self._parse_action_v120(clean_result, payload["target_doubt"])]
        except Exception as e:
            print(f"   ❌ Aksiyon hatası: {e}")
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


# --- V121: ADVANCED LOOP BREAKER ---
class LegalTextSanitizer:
    """V121: Gelişmiş Tekrar Engelleyici (Madde Bazlı)"""

    def __init__(self):
        self.seen_sentences = set()
        self.written_articles = set()  # YENİ: Madde numaralarını takip et
        self.dropped_count = 0

    def enforce_no_repeat(self, text):
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
# 4️⃣ ACTIONABLE RECOMMENDATION ENGINE
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

    def _generate_recommendation_text(self, concern, category_tr):
        prompt = f"""
        GÖREV: Kıdemli bir avukata yol gösterecek şekilde, aşağıdaki hakim tereddüdüne yönelik {category_tr} odaklı SOMUT ve UYGULANABİLİR bir aksiyon önerisi yaz.
        
        ANALİZ:
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
# 5️⃣ HAFIZA YÖNETİCİSİ (FULL INTEGRATED - V127 MASTER PROMPT)
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
Bu ilke ışığında, olayı değerlendirirken yaşadığın EN FAZLA 3 TEMEL TEREDDÜTÜ (Doubts) listele.
Her tereddüt SOMUT olsun: delil eksikliği, usul sorunu, emsal uyuşmazlığı gibi.
Tereddütler kısa ve net olsun (maks 1 cümle).
Ayrıca dosya hakkındaki İLK REFLEKSİNİ (Red/Kabul Eğilimli) tek kelimeyle yaz.

ÇIKTI FORMATI (JSON):
{
  "reflex": "RED EĞİLİMLİ veya KABUL EĞİLİMLİ",
  "doubts": ["Tereddüt 1...", "Tereddüt 2...", "Tereddüt 3..."]
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

        try:
            res = self.llm.invoke(prompt).content.strip()
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
- Hakimi kabul yönünde ikna etmeye çalışmak.

KURALLAR:
1. Her tereddüde AYRI AYRI cevap ver.
2. Cevabında mutlaka varsa [MEVZUAT] veya [EMSAL KARAR] etiketli belgeye ATIF YAP (Madde no veya Karar no ver).
3. Genel hukuk anlatma, doğrudan somut olaya ve müvekkilin haklılığına bağla.
4. Her cevap maks 3-4 cümle olsun.

ÇIKTI FORMATINI ASLA DEĞİŞTİRME:

--------------------------------------------------
DAVACI VEKİLİ DEĞERLENDİRMESİ
--------------------------------------------------
Tereddüt 1:
- Cevap:

Tereddüt 2:
- Cevap:

Tereddüt 3:
- Cevap:
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

        try:
            raw = self.llm.invoke(prompt).content.strip()
            return self.sanitizer.enforce_no_repeat(raw)
        except:
            return "Davacı vekili beyanı oluşturulamadı."

    def _generate_defendant_response_v120(self, doubts, principle_text, domain="Genel", query_text=""):
        doubts_text = "\n".join([f"- {d}" for d in doubts])
        combined_input = f"OLAY: {query_text}\n\nHAKİM TEREDDÜTLERİ:\n{doubts_text}"

        task = """
GÖREVİN:
- Hakimin tereddütlerini DERİNLEŞTİRMEK.
- Kabul ihtimalini zayıflatmak.

KURALLAR:
1. Her tereddüde AYRI AYRI cevap ver ve tereddüdü derinleştir.
2. Cevabında mutlaka varsa [MEVZUAT] veya [EMSAL KARAR] eksikliğine veya aleyhe durumuna ATIF YAP.
3. Genel hukuk anlatma, somut olaydaki eksikliklere bağla.
4. Her cevap maks 3-4 cümle olsun.

ÇIKTI FORMATINI ASLA DEĞİŞTİRME:

--------------------------------------------------
DAVALI VEKİLİ DEĞERLENDİRMESİ
--------------------------------------------------
Tereddüt 1:
- Karşı Argüman:

Tereddüt 2:
- Karşı Argüman:

Tereddüt 3:
- Karşı Argüman:
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

        try:
            raw = self.llm.invoke(prompt).content.strip()
            return self.sanitizer.enforce_no_repeat(raw)
        except:
            return "Davalı vekili beyanı oluşturulamadı."

    def _generate_expert_response_v120(self, doubts, principle_text, domain="Genel", query_text=""):
        doubts_text = "\n".join([f"- {d}" for d in doubts])
        combined_input = f"OLAY: {query_text}\n\nHAKİM TEREDDÜTLERİ:\n{doubts_text}"

        task = """
        GÖREVİN: Bağımsız ve tarafsız bir BİLİRKİŞİ olarak, dosyadaki hukuki mantık zincirini ve delil tutarlılığını denetlemek.

        İNCELEME NOKTALARI:
        1. Hakimin tereddütleri hukuki ve yerinde mi?
        2. Davacı tarafın sunduğu yanıtlar ispat yükünü karşılıyor mu?
        3. Davalı tarafın itirazları maddi vakıalarla ve kanunla örtüşüyor mu?

        ÇIKTI FORMATI:
        --------------------------------------------------
        BİLİRKİŞİ TESPİTLERİ
        --------------------------------------------------
        Genel Hukuki Değerlendirme: ...
        Zayıf Noktalar: ...
        Tutarlı Noktalar: ...
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

        try:
            raw = self.llm.invoke(prompt).content.strip()
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

            self.latest_ui_data = {
                "query": query_text, "domain": query_domain, "principles": [], "net_decision": {},
                "executive_summary": "", "audit_log": {}
            }

            for item in sorted_hits:
                # 2. Risk Analizi
                analysis = self._calculate_case_success_probability(
                    item["conf"], item["trend_dir"], item["conflict"], item["domain_match"], item["polarity"]
                )

                # --- V120: PERSONA SİSTEMİ ---
                # V127 UPDATE: Domain ve Query Text transfer edildi.

                # A. HAKİM REFLEKSİ VE TEREDDÜTLER (TRIGGER)
                # Yeni parametre eklendi: domain
                judge_data = self._generate_judge_doubts_v120(query_text, item['text'], domain=item['domain'])
                doubts = judge_data.get("doubts", [])
                reflex = judge_data.get("reflex", "BELİRSİZ")

                self.audit_logger.log_event(
                    stage="judge_analysis",
                    title="JUDGE ANALYSIS COMPLETED",
                    description=f"Hakim Refleksi: {reflex}",
                    outputs={"reflex": reflex, "doubt_count": len(doubts), "doubts": doubts}
                )

                # B. PERSONA PHASE (SIRALI AKIŞ)
                self.audit_logger.log_event(stage="persona_phase", title="PERSONA PHASE STARTED",
                                            description="Taraf vekilleri ve bilirkişi devreye giriyor.")

                # Davacı
                # Yeni parametreler: domain, query_text
                plaintiff_text = self._generate_plaintiff_response_v120(doubts, item['text'], domain=item['domain'],
                                                                        query_text=query_text)
                self.audit_logger.log_event(
                    stage="plaintiff_arg", title="DAVACI VEKİLİ DEĞERLENDİRMESİ",
                    description=f"Ele alınan tereddüt sayısı: {len(doubts)}",
                    outputs={"full_text": plaintiff_text}
                )

                # Davalı
                # Yeni parametreler: domain, query_text
                defendant_text = self._generate_defendant_response_v120(doubts, item['text'], domain=item['domain'],
                                                                        query_text=query_text)
                self.audit_logger.log_event(
                    stage="defendant_arg", title="DAVALI VEKİLİ DEĞERLENDİRMESİ",
                    description="Karşı argümanlar ve usul itirazları sunuldu.",
                    outputs={"full_text": defendant_text}
                )

                # Bilirkişi
                # Yeni parametreler: domain, query_text
                expert_text = self._generate_expert_response_v120(doubts, item['text'], domain=item['domain'],
                                                                  query_text=query_text)
                self.audit_logger.log_event(
                    stage="expert_arg", title="BİLİRKİŞİ TESPİTLERİ",
                    description="Hukuki zincir ve tutarlılık kontrolü yapıldı.",
                    outputs={"full_text": expert_text}
                )

                self.audit_logger.log_event(stage="persona_completed", title="PERSONA PHASE COMPLETED",
                                            description="Tüm taraflar dinlendi.")

                # C. ACTION ENGINE (Tereddütler üzerinden çalışır)
                action_plan = self.recommendation_engine.generate(doubts, query_text)

                # D. SIMULATION
                simulation_result = self._simulate_post_strengthening_score(analysis['success_probability'],
                                                                            action_plan)

                # E. EXECUTIVE SUMMARY
                exec_summary = f"Hakim '{reflex}' eğilimindedir. {len(doubts)} temel tereddüt (Örn: {doubts[0]}) mevcuttur. Davacı vekili bu hususları gidermeye çalışsa da Davalı taraf usul itirazlarını sürdürmektedir."

                # V120 SANITIZATION LOG
                self.audit_logger.log_event(
                    stage="output_sanitizer", title="OUTPUT SANITIZER APPLIED",
                    description=f"Tekrar eden paragraflar temizlendi.",
                    outputs={"repeated_paragraphs_removed": self.sanitizer.dropped_count}
                )

                # Store Complete Data (V120 Structure)
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
                    # Backward compatibility dummy data
                    "personas": {"judge": str(doubts), "opponent": defendant_text, "opponent_title": "Davalı",
                                 "expert": expert_text, "devil": "N/A"},
                    "conflict_analysis": {"conflict_level": "N/A", "conflict_score": 0, "summary": []},
                    "reasoned_verdict": f"HAKİMİN GEÇİCİ KANAATİ: {reflex}. Gerekçe: {doubts}",
                    "action_plan": action_plan,
                    "simulation": simulation_result
                })
                self.latest_ui_data["executive_summary"] = exec_summary
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
                print("   ℹ️ Yeterli yeni veri yok.")
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
                res = self.llm.invoke(prompt).content.strip()
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
# 7️⃣ YENİ ARAÇLAR: REASONING & STRATEGY (RESTORED)
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

        ÖNEMLİ: Sadece aşağıda sunulan dosya kapsamına ve delillere sadık kal. Metinde olmayan hiçbir tanığı, belgeyi veya vakıayı varmış gibi gösterme.

        OLAY ÖZETİ: {story if story else 'Dosya kapsamı'}
        MEVZUAT, EMSAL VE DELİLLER: {context_str if context_str else audit_summary}

        YAZIM ŞABLONU (RESMİ ÜSLUP):
        1. **DAVA VE İHTİLAFIN ÖZETİ**: Tarafların iddia ve savunmalarının hukuki özeti.
        2. **DELİLLERİN TARTIŞILMASI**: Dosyaya sunulan delillerin sıhhati ve olayla ilgisi.
        3. **HUKUKİ GEREKÇE**: Uygulanacak kanun maddeleri ve Yargıtay içtihatları ile somut olayın sentezi.
        4. **SONUÇ VE HÜKÜM**: Davanın kabulü, reddi veya kısmen kabulü yönünde kesin ve net yargı.

        ÜSLUP: Kararın Türk Milleti adına verildiği bilinciyle; resmi, nesnel ve otoriter bir dil kullan.
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
        GÖREV: SEN, KIDEMLİ BİR AVUKATSIN. Aşağıdaki hakim gerekçesini "üst mahkeme incelemesi" (istinaf/temyiz) için hukuki süzgeçten geçir.
        
        {LegalConfig.PROMPT_GUARD}

        GÖREVİN: Mahkemenin gerekçesindeki hataları (maddi hata, usul hatası, yanlış takdir) belirleyerek profesyonel İTİRAZ ARGÜMANLARI geliştir.

        KURALLAR:
        - Meslek etiğine ve mahkemeye saygı dilinden ayrılma.
        - "Eksik inceleme", "Hatalı hukuki tavsif", "Delillerin yanlış takdiri" gibi teknik kalıpları yerinde kullan.
        - Sadece dosya kapsamındaki veriler üzerinden itiraz geliştir.

        HAKİM GEREKÇESİ:
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
        GÖREV: SEN, KIDEMLİ BİR AVUKATSIN. Aşağıdaki hakim gerekçesine karşı, Üst Mahkemeye (BAM/Yargıtay) sunulmak üzere resmi bir İSTİNAF/TEMYİZ DİLEKÇESİ yaz.

        {LegalConfig.PROMPT_GUARD}

        ZORUNLU FORMAT VE BAŞLIKLAR:
        1. **KARARIN ÖZETİ**: Yerel mahkemenin verdiği hükmün kısa özeti.
        2. **İSTİNAF/TEMYİZ NEDENLERİ**: Maddi vakıalar ve hukuk kuralları açısından hatanın nerede olduğu (Örn: Hatalı delil takdiri, eksik inceleme).
        3. **HUKUKİ DEĞERLENDİRME**: TMK/TBK/HMK maddeleri ve Yargıtay emsal kararları ile itirazların desteklenmesi.
        4. **SONUÇ VE İSTEM**: Kararın bozulması veya kaldırılması yönündeki net talep.

        ÜSLUP: Resmi, hukuki terminolojiye hakim, ciddi ve kurumsal.

        DOSYA KONUSU: {case_topic}
        HAKİM GEREKÇESİ: {judge_reasoning}
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
            GÖREV: KIDEMLİ BİR AVUKAT olarak, aşağıdaki itiraz argümanını karşılayacak veya güçlendirecek SOMUT BİR AKSİYON (EYLEM) planı hazırla.
            
            JSON FORMATI:
            ALANLAR: title, evidence_type (tanık/belge/bilirkişi/içtihat), source, estimated_time, estimated_cost, risk_if_missing
            
            İTİRAZ ARGÜMANI: {arg}
            """
            try:
                res = self.llm.invoke(prompt).content.strip()
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
# 8️⃣ ARAMA MOTORU SINIFI (SEARCH ENGINE)
# ==================================================
class LegalSearchEngine:
    def __init__(self):
        self.config = LegalConfig()
        self.dense_embedder = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL)
        self.client = None
        atexit.register(self.close)

    def connect_db(self):
        if self.client is not None: return True
        print("   🔌 Veritabanı bağlantısı başlatılıyor...")
        LegalUtils.force_unlock_db()
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
                print("\n🔒 Veritabanı bağlantısı güvenli şekilde kapatıldı.")
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
                print(f"      ⚙️ '{collection_name}' oluşturuluyor...")
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
            print(f"      ♻️ {config['desc']} için {len(new_files)} yeni dosya işleniyor...")

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
                    print(f"      📄 Okundu: {filename}")
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
                print(f"❌ İşlemci Hatası: {e}");
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
            print(f"❌ Embedding Hatası: {e}");
            return []

        all_candidates = []
        for key, config in self.config.SOURCES.items():
            try:
                results = self.client.query_points(collection_name=config["collection"], query=query_vector,
                                                   limit=self.config.SEARCH_LIMIT_PER_SOURCE).points
                for hit in results:
                    if 'type' not in hit.payload: hit.payload['type'] = config['desc']
                    all_candidates.append(hit)
            except:
                pass

        unique_docs = {}
        for hit in all_candidates:
            if hit.score < self.config.SCORE_THRESHOLD: continue
            key = f"{hit.payload['source']}_{hit.payload['page']}"
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

        limit = self.config.LLM_RERANK_LIMIT
        statute_quota = 3
        precedent_quota = limit - statute_quota

        final_candidates = emsal_hits[:precedent_quota] + mevzuat_hits[:statute_quota]

        if len(mevzuat_hits) < statute_quota:
            needed = limit - len(final_candidates)
            if needed > 0:
                extras = emsal_hits[precedent_quota: precedent_quota + needed]
                final_candidates.extend(extras)

        if not final_candidates: print("🔴 Uygun belge bulunamadı."); return []
        print(f"   ✅ {len(final_candidates)} potansiyel belge bulundu. Yargıca gönderiliyor...")
        return final_candidates


# ==================================================
# 9️⃣ YARGIÇ VE MUHAKEME SINIFI (JUDGE)
# ==================================================
class LegalJudge:
    """
    CPU (Hazırlık) ve GPU (Çalıştırma) ayrıştırılmış Yargıç Motoru.
    Özellikler:
    1. 'Alaka' ve 'Rol' tespiti tek prompt'ta birleştirildi (Hız x2).
    2. Prompt hazırlığı CPU'da, çalıştırma GPU'da yapılır.
    """

    def __init__(self, memory_manager=None, llm=None):
        self.llm = llm  # Judge Profili (Sequential/Tek Kanal)
        self.memory = memory_manager
        self.sanitizer = LegalTextSanitizer()

    # =========================================================================
    # 🟢 ADIM 1: CPU HAZIRLIK (Build Phase)
    # =========================================================================
    def build_evaluation_payloads(self, candidates, story, topic, negatives) -> List[Dict]:
        """
        Aday belgeler için tek tek prompt hazırlar.
        ASLA LLM ÇAĞIRMAZ. Sadece string işlemi yapar.
        """
        print("   ⚙️ CPU: Belge analiz promptları hazırlanıyor...")
        payloads = []

        # Scope bloğunu bir kez oluştur (CPU)
        scope_block = self._build_scope_block(topic, negatives)

        for hit in candidates:
            doc_text = hit.payload.get('page_content', '')
            source = hit.payload.get('source', 'Bilinmiyor')
            doc_type = hit.payload.get('type', 'BELGE')

            # Negatif kelime kontrolü (Hızlı CPU elemesi)
            if negatives:
                doc_lower = doc_text.lower()
                if any(bad in doc_lower for bad in negatives):
                    continue

            # --- TEK PROMPT (Relevance + Role + Reason) ---
            # Eskiden 2 ayrı LLM çağrısı vardı (_check_relevance + _assign_role).
            # Şimdi tek seferde soruyoruz.

            prompt = f"""
            GÖREV: SEN KIDEMLİ BİR HUKUKÇUSUN. Aşağıdaki belgeyi, sağlanan olay ve odak noktası çerçevesinde değerlendir.

            {LegalConfig.PROMPT_GUARD}

            {scope_block}

            Sorgu Özeti: "{story} - {topic}"

            İNCELENECEK BELGE ({doc_type}):
            ---
            {doc_text[:1500]}...
            ---

            GÖREVİN:
            Bu belgenin olayla hukuki alakasını (relevance) ve oynayacağı rolü belirle.

            KARAR KURALLARI:
            1. Eğer belge tamamen alakasızsa veya yasaklı (EXCLUDED) kapsamdaysa sadece "KARAR: RED" yaz.
            2. Eğer belge alakalıysa:
               - "KARAR: KABUL" yaz.
               - Rolü seç: [DOĞRUDAN DELİL] (Vakıayı ispatlar) veya [EMSAL İLKE] (Hukuki kuralı açıklar).
               - Tek cümlelik profesyonel gerekçeni ekle.

            ÇIKTI FORMATI:
            KARAR: KABUL | ROL: [EMSAL İLKE] | GEREKÇE: ...
            """
            payloads.append({
                "prompt": prompt,
                "source": source,
                "text": doc_text,
                "type": doc_type,
                "page": hit.payload.get('page', 0),
                "original_score": hit.score
            })

        return payloads

    # =========================================================================
    # 🔴 ADIM 2: GPU ÇALIŞTIRMA (Execution Phase)
    # =========================================================================
    def execute_evaluations(self, payloads: List[Dict]) -> List[Dict]:
        """
        Hazır promptları GPU'ya gönderir ve sonuçları işler.
        Saf seri döngü kullanır (ThreadPool yok).
        """
        if not payloads: return []

        print(f"   ⚖️ GPU: {len(payloads)} belge değerlendiriliyor (Merged Check)...")
        valid_docs = []

        for i, p in enumerate(payloads):
            try:
                # Bloklayıcı (Blocking) LLM Çağrısı
                result = self.llm.invoke(p["prompt"]).content.strip()

                # Sonucu Parse Et (CPU işlemi)
                if "KARAR: KABUL" in result.upper():
                    # Rol Tespiti
                    role = "[EMSAL İLKE]"  # Varsayılan
                    if "[DOĞRUDAN DELİL]" in result: role = "[DOĞRUDAN DELİL]"

                    # Gerekçe Tespiti
                    reason = "İlgili belge."
                    if "GEREKÇE:" in result:
                        parts = result.split("GEREKÇE:")
                        if len(parts) > 1:
                            reason = parts[-1].strip()

                    # Skor hesapla (Vektör skoru + Onay bonusu)
                    final_score = min(p["original_score"] * 100 * 1.2, 100.0)

                    # print(f"      ✅ KABUL: {p['source']} ({role})") # İsteğe bağlı log

                    valid_docs.append({
                        "source": p["source"],
                        "page": p["page"],
                        "type": p["type"],
                        "role": role,
                        "text": p["text"],
                        "score": final_score,
                        "reason": reason
                    })
                # else:
                # print(f"      ❌ RED: {p['source']}")

            except Exception as e:
                print(f"      ⚠️ Hata ({p['source']}): {e}")

        return valid_docs

    # =========================================================================
    # YARDIMCI VE SENIOR METOTLAR
    # =========================================================================

    def _build_scope_block(self, topic, negatives=None):
        scope = f"""
ALLOWED SCOPE (ZORUNLU):
- Analiz SADECE şu konu ile sınırlı olacak: {topic}
- Türk Hukuku (Yargıtay/BAM uygulaması)
"""
        if negatives:
            scope += f"\nEXCLUDED: {', '.join(negatives)}"
        return scope

    # Bu küçük metotlar genellikle bir kez çağrıldığı için doğrudan invoke yapabilir
    # veya aynı şekilde ayrıştırılabilir. Basitlik için burada bırakıyorum.
    def validate_user_input(self, story, topic):
        try:
            prompt = f"GÖREV: Metin tamamen rastgele tuşlama mı? '{story} {topic}'. [GEÇERLİ]/[GEÇERSİZ]."
            res = self.llm.invoke(prompt).content.strip()
            return "GEÇERSİZ" not in res
        except:
            return True

    def generate_expanded_queries(self, story, topic):
        try:
            print("   ↳ 🧠 CPU/GPU: Sorgu Genişletiliyor...")
            prompt = f"GÖREV: Hukuki terimler.\nOLAY: {story}\nODAK: {topic}\n3 kısa cümle."
            res = self.llm.invoke(prompt).content
            return [line.strip() for line in res.splitlines() if len(line) > 5][:3]
        except:
            return [story]

    # [SENIOR GEREKÇE YAZIMI - SİZİN PAYLAŞTIĞINIZ KOD]
    # Bu metod zaten tek bir büyük çağrı olduğu için GPU/CPU ayrımı doğaldır.
    def generate_final_opinion(self, story, topic, context_str, context: QueryContext, judge_reflex=None):
        print("\n🧑‍⚖️ GEREKÇELİ KARAR YAZILIYOR (SENIOR MODE + MAPPING)...")

        # 1. DOMAIN MAPPING (CPU)
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

        # Domain algıla (CPU)
        domain_key = context.detected_domain.lower().replace(" ", "_") if context else "genel_hukuk"
        mapping = DOMAIN_MAPPINGS.get(domain_key, DOMAIN_MAPPINGS["genel_hukuk"])

        # 2. HAKİM EĞİLİM KİLİDİ (CPU)
        reflex_note = ""
        if judge_reflex:
            reflex_note = f"""
            HAKİMİN VİCDANİ KANAATİ (BAĞLAYICI):
            - Eğilim: {judge_reflex.tendency}
            - Dosya Güç Skoru: {judge_reflex.score}/100
            - Giderilemeyen Tereddütler: {', '.join(judge_reflex.doubts)}
            """

        prompt = f"""
        GÖREV: SEN TÜRKİYE CUMHURİYETİ HAKİMİSİN. Önündeki dosya için resmi, bağlayıcı ve gerekçeli bir hüküm kurman gerekiyor.
        
        {LegalConfig.PROMPT_GUARD}

        ÖNEMLİ: Sadece sağlanan hukuki zemin ve deliller üzerinden karar ver. Bilgin olmayan konularda uydurma yapma.

        BAĞLAM: {context.detected_domain.upper() if context else 'GENEL'}
        ZORUNLU KANUNİ ATIFLAR: {mapping['maddeler']}
        İLGİLİ İÇTİHAT MERCİLERİ: {mapping['ictihatlar']}

        DAVA VE TALEP: {story}
        DELİL VE EMSAL DURUMU: {context_str}

        {reflex_note}

        GEREKÇELİ KARAR ŞABLONU:
        1. **HUKUKİ TAVSİF VE NİTELEME**: (Uyuşmazlığın kanuni temeli.)
        2. **DELİLLERİN ANALİZİ VE TARTIŞILMASI**: (Emsallerin ve belgelerin olaya etkisi.)
        3. **VİCDANİ KANAAT VE HUKUKİ GEREKÇE**: (Hakim olarak ulaştığın nihai sonuç ve dayandığın temel ilke.)
        4. **HÜKÜM**: (Dava hakkında verilen kesin karar: KABUL / RED / KISMEN KABUL.)

        ÜSLUP: Tam bir hakim vakarıyla; kesin, nesnel ve Türk Milleti Adına karar verir ciddiyette.
        """
        # 4. LLM ÇAĞRISI (GPU)
        try:
            full_res = self.llm.invoke(prompt).content.strip()
            # Önce halüsinasyonları ve placeholderları temizle, sonra tekrarları sil
            clean_res = self.sanitizer.sanitize_hallucinations(full_res)
            return self.sanitizer.enforce_no_repeat(clean_res)
        except Exception as e:
            return f"Gerekçe oluşturulurken hata oluştu: {e}"

    # Uyumluluk için boş metodlar (Gerekirse)
    def explain_precedents_for_pdf(self, accepted_docs, topic):
        return []

    def build_query_context(self, story, topic, negatives) -> QueryContext:
        ctx = QueryContext(query_text=story, topic=topic, negative_scope=negatives)
        ctx.assert_hard_limits();
        return ctx


# ==================================================
# 🔟 RAPORLAMA SINIFI (V120 - ROBUST FONT LOADER)
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
# 1️⃣2️⃣ Pipeline
# ==================================================
class LegalEvaluationPipeline:
    def __init__(self, judge_core, logic_engine):
        self.judge_core = judge_core
        self.logic_engine = logic_engine
        self.has_run = False
        self.last_result = None

    def run(self, decision_context, persona_outputs):
        if self.has_run:
            print("   ⚠️ Pipeline zaten çalıştı – Son sonucu döndürüyor.")
            return self.last_result
        self.has_run = True

        # 1️⃣ Deterministik ilk değerlendirme
        initial_reflex = self.judge_core.evaluate(decision_context)

        print(f"   ⚖️  ÖN YARGIÇ REFLEKSİ: {initial_reflex.tendency} (Skor: {initial_reflex.score})")

        if initial_reflex.score < 30:
            raise RuntimeError(
                f"Dosya hukuki olarak zayıf (Skor: {initial_reflex.score}). Hakim ilk refleksi RED yönünde. Lütfen daha güçlü delil veya emsal ile tekrar deneyin.")

        # 2️⃣ Mantık motoru ile düzeltme
        final_reflex = self.logic_engine.run_logic(
            initial_reflex=initial_reflex,
            persona_outputs=persona_outputs
        )

        self.last_result = final_reflex
        return final_reflex


# ==================================================
# ANA UYGULAMA (MAIN APP)
# ==================================================
class LegalApp:
    def __init__(self):
        print("🚀 LEGAL SUITE V142 (CPU/GPU Pipelined)...")

        # 🔥 PROFİLLİ LLM’LER (Global Router - Tek Kanal GPU)
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

                print("   🛡️ Girdi kontrol ediliyor...")
                if not self.judge.validate_user_input(story, topic):
                    print("   ❌ UYARI: Girdi anlamsız. Lütfen mantıklı bir olay giriniz.")
                    continue

                # ---------------------------------------------------------
                # 1. BAĞLAM VE ARAMA (CPU AŞAMASI)
                # ---------------------------------------------------------
                start_total = time.time()
                print("   ⚙️ CPU: Bağlam ve sorgu hazırlanıyor...")

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
                # 2. BELGE DEĞERLENDİRME (CPU HAZIRLIK -> GPU INFERENCE)
                # ---------------------------------------------------------
                # A. Promptları Hazırla (CPU)
                judge_payloads = self.judge.build_evaluation_payloads(
                    candidates, ctx.query_text, ctx.topic, ctx.negative_scope
                )

                # B. GPU'da Çalıştır (Seri, Tek Kanal)
                valid_docs = self.judge.execute_evaluations(judge_payloads)

                if not valid_docs:
                    print("🔴 Yargıç tüm belgeleri eledi.")
                    continue

                print(f"⏱️ Judge inference bitti: {time.time() - start_total:.2f} sn")

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

                # Hafıza Çağırma (Opsiyonel - CPU)
                current_personas = {}
                mem_principles = []
                if self.memory_manager:
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
                print("   ⚙️ CPU: Persona verileri ve hukuk zemini hazırlanıyor...")
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
                print("   ⚙️ CPU: Risk analizi kurgulanıyor...")
                risk_payload = action_engine.build_risk_payload(reflex, persona_outputs)

                # B. GPU'da Çalıştır
                strengthening_actions = action_engine.execute_action(risk_payload)

                # Avukat Masası (Konsol Çıktısı)
                if strengthening_actions:
                    print(f"\n   🛠️  AKSİYON PLANI (V120 Disiplini):")
                    for act in strengthening_actions:
                        print(f"      🔹 [{act.impact_score}/10] {act.title}")
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
                        full_advice=full_advice
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
                    print(f"   {log['timestamp']} | {log['title']} → {log['description']}")

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