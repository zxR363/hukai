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
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support
from dataclasses import dataclass, field
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

# 🔨 Commit 5.3: Query Context (Single Source of Truth)
@dataclass
class QueryContext:
    """
    Sistemde TEK bağlayıcı bağlam nesnesi.
    Tüm modüller yalnızca bunu referans alır.
    """
    # Kullanıcı girdisi
    query_text: str

    # Hukuki bağlam
    topic: str
    detected_domain: str  # örn: "miras", "icra", "ceza"

    # Kapsam sınırları
    negative_scope: List[str]
    allowed_sources: List[str] = None

    # Sistem içi bayraklar
    allow_analogy: bool = False
    allow_speculation: bool = False
    allow_soft_language: bool = False

    # 🆕 EKLENECEK SATIR (Guard Bayrağı)
    judge_evaluated: bool = False

    def assert_hard_limits(self):
        """
        Hukuki güvenlik kemeri.
        """
        if self.allow_speculation:
            raise ValueError("Speculation is forbidden in legal analysis.")

        if self.allow_analogy:
            raise ValueError("Analogy is forbidden unless explicitly enabled.")


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
@dataclass
class StrengtheningAction:
    title: str
    description: str
    related_doubt: str
    impact_score: int  # 1–10 arası katkı puanı


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
ZORUNLU YAZIM VE AKIL YÜRÜTME KURALLARI:

1. SADECE verilen olay, scope ve hukuki bağlam içinde kal.
2. Genel hukuk bilgisi, öğretici anlatım veya akademik açıklama YAPMA.
3. “Genel olarak”, “çoğunlukla”, “doktrinde” gibi belirsiz ifadeler KULLANMA.
4. Aynı hukuki ilkeyi veya TMK/Yargıtay maddesini BİR KEZ açıkla.
5. Aynı düşünceyi farklı kelimelerle TEKRAR ETME.
6. Somut olayla bağlantısı olmayan hiçbir bilgi EKLEME.
7. Emsal yoksa uydurma; belirsizlik varsa AÇIKÇA belirt.
8. Değer yargısı, ahlaki yorum, sosyal politika yorumu YAPMA.
9. “Bu durumda karar verilmelidir” gibi HÜKÜM KURAN ifadeler kullanma.
10. Hakim, avukat veya bilirkişi rolü dışında düşünme.
11. Çıktı, gerçek bir mahkeme dosyasına girebilecek ciddiyette olsun.
12. Bu kuralların dışına çıkma; çıktıyı bu kurallara göre DENETLE.
13.Her belge yalnızca bir kez özetlenir.Özet, sorgudaki somut olayla doğrudan bağ kurmak zorundadır.
"Bu belge, sorgudaki [X] durumuna şu şekilde uygulanır: ..." formatı zorunludur.
14.Belge → Hukuki İlke → Somut Olay → Dosyaya Etki zinciri kurulmadan belge kullanılamaz.
15. Subjektif kelimeler ("benzetebilirsiniz", "olabilir", "gibi") KULLANMA; her atıf SOMUT olsun ("Yargıtay 14. HD 2015/2278 E. kararında şöyle belirtilmiştir: ...").
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


# 🔨 Commit 5.6: Persona Engine (Kontrollü LLM)
class PersonaEngine:
    """
    LLM kontrollü persona simülasyonu.
    Hakimin tereddütlerine cevap üretir.
    """

    def __init__(self, llm):
        self.llm = llm
        self.current_doubts = []

    def run(
            self,
            context: QueryContext,
            decision_context: DecisionContext,
            judge_reflex: JudgeReflex
    ) -> List[PersonaResponse]:

        self.current_doubts = judge_reflex.doubts
        if not self.current_doubts:
            # Tereddüt yoksa standart bir başlangıç ata
            self.current_doubts = ["Dosyanın esasına ilişkin genel delil durumu", "Hukuki tavsif"]

        print(f"   🗣️  Persona Tartışması Başlatılıyor ({len(self.current_doubts)} Tereddüt)...")
        responses = []

        responses.append(
            self._invoke_persona(
                role="DAVACI VEKİLİ",
                instruction="Hakimin tereddütlerini gider, davanın kabulü için argüman üret."
            )
        )

        responses.append(
            self._invoke_persona(
                role="DAVALI VEKİLİ",
                instruction="Hakimin tereddütlerini derinleştir, davanın reddi için itiraz et."
            )
        )

        responses.append(
            self._invoke_persona(
                role="BİLİRKİŞİ",
                instruction="Tereddütlerin hukuki tutarlılığını ve delil zincirini denetle."
            )
        )

        return responses

    def _invoke_persona(self, role: str, instruction: str) -> PersonaResponse:
        prompt = f"""
        ROL: {role}
        BAĞLAM: Türk Hukuku.
        {LegalConfig.PROMPT_GUARD}

        GÖREV:
        {instruction}

        HAKİMİN SOMUT TEREDDÜTLERİ:
        {self._format_doubts()}

        SINIRLAR:
        - Yeni hukuki kural üretme.
        - Hakim kararını değiştirmeye çalışma (Sadece ikna et/eleştir).
        - Skor veya oran verme.
        - Sadece yukarıdaki tereddütlere odaklan.

        ÇIKTI:
        - Net, hukuki dilde, maksimum 2 paragraf.
        """

        try:
            result = self.llm.invoke(prompt).content.strip()
        except:
            result = f"{role}: Beyan oluşturulamadı."

        return PersonaResponse(
            role=role,
            response=result,
            addressed_doubts=self.current_doubts
        )

    def _format_doubts(self):
        return "\n".join(f"- {d}" for d in self.current_doubts)


# 🔨 Commit 5.7: Action Engine
class ActionEngine:
    """
    Hakim tereddütlerini azaltmaya yönelik
    somut hukuki aksiyonlar üretir.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(
            self,
            judge_reflex: JudgeReflex,
            persona_outputs: List[PersonaResponse]
    ) -> List[StrengtheningAction]:

        if not judge_reflex.doubts:
            return []

        actions = []

        for doubt in judge_reflex.doubts:
            action = self._generate_action(doubt, persona_outputs)
            if action:
                actions.append(action)

        return actions

    def _generate_action(
            self,
            doubt: str,
            persona_outputs: List[PersonaResponse]
    ) -> StrengtheningAction:

        persona_context = "\n\n".join(
            f"{p.role}: {p.response}"
            for p in persona_outputs
            # Eğer persona cevabında bu doubt geçiyorsa al, yoksa hepsini al (basit eşleşme)
            if True
        )

        prompt = f"""
        HAKİM TEREDDÜDÜ:
        {doubt}

        PERSONA DEĞERLENDİRMELERİ:
        {persona_context}

        GÖREV:
        Bu tereddüdü azaltmak için yapılabilecek
        TEK ve SOMUT hukuki aksiyonu yaz.

        SINIRLAR:
        - Tavsiye tonu kullanma
        - Genel laf üretme
        - En fazla 3 cümle

        FORMAT:
        Başlık:
        Açıklama:
        Etki Puanı (1-10):
        """

        try:
            result = self.llm.invoke(prompt).content
            return self._parse_action(result, doubt)
        except:
            return None

    def _parse_action(self, text: str, doubt: str) -> StrengtheningAction:
        lines = text.splitlines()

        title = "Ek Delil Sunumu"
        description = "İlgili hususta ek delil sunulmalıdır."
        impact = 5

        for line in lines:
            if "Başlık" in line:
                parts = line.split(":", 1)
                if len(parts) > 1: title = parts[1].strip()
            elif "Açıklama" in line:
                parts = line.split(":", 1)
                if len(parts) > 1: description = parts[1].strip()
            elif "Etki" in line:
                try:
                    # Sadece rakamları al
                    impact = int("".join(filter(str.isdigit, line)))
                    # 10'dan büyükse (örn 810) son basamağı al veya 10 yap
                    if impact > 10: impact = 5
                except:
                    impact = 5

        return StrengtheningAction(
            title=title,
            description=description,
            related_doubt=doubt,
            impact_score=impact
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
SEN BİR {hakim_rolu} OLARAK KARAR GEREKÇESİ YAZIYORSUN.
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
{LegalConfig.PROMPT_GUARD}

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
        BU SINIF KARAR VERMEZ.
        JudgeCore tarafından üretilmiş eğilimi,
        hukuki rapor diline çevirir.
    """

    def __init__(self, memory_manager=None):
        # V120: Global Config Kullanımı
        self.llm = ChatOllama(
            model=LegalConfig.LLM_MODEL,
            temperature=LegalConfig.LLM_CONFIG["temperature"],
            top_p=LegalConfig.LLM_CONFIG["top_p"],
            # Diğer parametreler LangChain entegrasyonuna göre kwargs olarak geçilebilir
            # ancak temel olarak temp ve top_p yeterlidir.
        )
        self.memory = memory_manager
        self.sanitizer = LegalTextSanitizer()

    # 🔨 Commit 5.3: Build Query Context (Single Source)
    def build_query_context(self, story, topic, negatives) -> QueryContext:
        """
        Ham kullanıcı girdilerini alır, hukuk alanını tespit eder ve
        tek bir QueryContext nesnesi olarak paketler.
        """
        # Hukuk Alanı Tespiti (Memory varsa oradan, yoksa basitçe 'Genel')
        domain = "Genel"
        if self.memory:
            # Domain tespiti için memory_manager içindeki fonksiyonu kullanıyoruz
            domain = self.memory._detect_domain_from_query(f"{story} {topic}")

        ctx = QueryContext(
            query_text=story,
            topic=topic,
            detected_domain=domain,
            negative_scope=negatives,
            allowed_sources=["mevzuat", "emsal"],
            allow_analogy=False,
            allow_speculation=False,
            allow_soft_language=False
        )

        # Güvenlik kemerini bağla
        ctx.assert_hard_limits()

        return ctx

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

    # -------------------------------------------------------------------------
    # DÜZELTME 1: JudgeReflex parametresi eklendi ve Prompt kısıtlandı
    # -------------------------------------------------------------------------
    def generate_final_opinion(self, story, topic, context_str, judge_reflex=None):
        print("\n🧑‍⚖️  AVUKAT YAZIYOR (V120: Final Output)...")

        # Eğer JudgeCore sonucu geldiyse prompt'a gömüyoruz
        decision_lock = ""
        if judge_reflex:
            decision_lock = f"""
        🛑 KESİN TALİMAT (JUDGE CORE LOCK):
        Sistem tarafından yapılan matematiksel analiz sonucunda:
        1. HAKİM EĞİLİMİ: "{judge_reflex.tendency}" olarak tespit edilmiştir.
        2. DOSYA GÜÇ SKORU: {judge_reflex.score}/100
        3. TESPİT EDİLEN TEREDDÜTLER: {', '.join(judge_reflex.doubts)}

        GÖREVİN:
        YENİDEN HÜKÜM KURMAK DEĞİL, YUKARIDAKİ "{judge_reflex.tendency}" HAKİM EĞİLİMİNİ
        hukuki riskler ve gerekçeler çerçevesinde değerlendir.
        Bu bir nihai karar değildir..
        Analizini bu kararı destekleyecek veya bu kararın risklerini açıklayacak şekilde yap.
        
        ⚠️ UYARI:
        Bu metin mahkeme kararı değildir.
        Hakimin ilk değerlendirme eğilimini yansıtan bir analiz raporudur.
        """

        system_content = f"""SEN, TÜRK MAHKEMESİNDE GÖREVLİ BİR HAKİM RAPORTÖRÜSÜN.
HÜKMÜ SEN VERMİYORSUN; VERİLMİŞ HÜKMÜN GEREKÇESİNİ YAZIYORSUN.
{LegalConfig.PROMPT_GUARD}

🛑 KRİTİK VE ZORUNLU KURAL:
BU SİSTEM SADECE TÜRKÇE ÇALIŞIR. 

{decision_lock}

HER NE OLURSA OLSUN ÇIKTIYI SADECE VE SADECE **TÜRKÇE** DİLİNDE VER.
(RESPONSE MUST BE ONLY IN TURKISH LANGUAGE. DO NOT USE CHINESE OR ENGLISH.)

Görevin:
- Tarafları savunmak DEĞİL
- JudgeCore tarafından belirlenen eğilim doğrultusunda
  RED riskinin nedenleri ve azaltma yollarını değerlendir.

NORMLAR HİYERARŞİSİ (ZORUNLU):
- [MEVZUAT] etiketli metinler KANUN maddesidir (TMK, BK vb.). Bunları kesin kural olarak sun.
- [EMSAL KARAR] etiketli metinler YARGITAY uygulamasıdır. Bunları "yorum ve uygulama" olarak sun.

ÖN KABULLER:
1. Veraset ilamı çekişmesiz yargı işidir.
2. Çekişmesiz yargı kararları maddi anlamda kesin hüküm oluşturmaz.
3. Hakim her zaman önce RED ihtimalini değerlendirir.
4. Usul eksikliği varsa ESASA GİRİLMEZ.
5. Analiz bölümünde en fazla 5 belge kullan.
6. Her belge en fazla 3 cümleyle özetlenir.
7. Aynı belge ikinci kez yazılamaz.


SANA SAĞLANAN BELGELER ETİKETLİDİR:
- [MEVZUAT]
- [EMSAL KARAR]

BELGE DIŞINA ÇIKMA.
YENİ EMSAL UYDURMA.
GENEL HUKUK ANLATISI YAPMA.

----------------------------------------------------------------
AŞAMA 1 — JUDGE CORE DEĞERLENDİRMESİNİN HUKUKİ OKUMASI
----------------------------------------------------------------
UYARI:
Bu aşamada YENİ bir değerlendirme yapma.
Sadece JudgeCore tarafından tespit edilen tereddütleri hukuki dile çevir.
Yeni tereddüt ekleme.

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

        system_content += "\nSOMUT ATIF ZORUNLULUĞU: Her emsal/mevzuat atfını karar numarasıyla yap (örneğin, 'Yargıtay 14. HD 2015/2278 E. kararında ıskatın veraset ilamında şerh düşülmesi gerektiği belirtilmiştir'). allow_soft_language=False: Subjektif ifadeler yasak."

        user_content = f"""Aşağıdaki "DELİLLER" listesinde sunulan belgeleri kullanarak olayı analiz et.
OLAY: "{story}"
ODAK: "{topic}"
DELİLLER:
{context_str}
ANALİZİ BAŞLAT (TÜRKÇE):"""

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]

        full_res = ""
        for chunk in self.llm.stream(messages):
            c = chunk.content;
            full_res += c;
            print(c, end="", flush=True)
        print("\n")

        # V120 SANITIZATION
        cleaned_res = self.sanitizer.enforce_no_repeat(full_res)

        if judge_reflex and not _contains_decision(cleaned_res, judge_reflex.tendency):
            cleaned_res = (
                    f"⚠️ JUDGE CORE EĞİLİMİ: {judge_reflex.tendency}\n\n"
                    + cleaned_res
            )

        return cleaned_res


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

    def run(self, decision_context, persona_outputs):
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

        return final_reflex



# ==================================================
# ANA UYGULAMA (MAIN APP)
# ==================================================
class LegalApp:
    def __init__(self):
        print("🚀 LEGAL SUITE V128 (Precedent Layer Added)...")
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
        # [YENİ] Mantık Motorunu Başlat
        self.logic_engine = LegalDecisionLogic()

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

                # 🔨 Commit 5.3: Single Source of Truth
                # Artık dağınık değişkenler yerine "QueryContext" nesnesi oluşturuyoruz.
                ctx = self.judge.build_query_context(story, topic, negatives)
                print(f"   ✓ Bağlam Oluşturuldu: {ctx.detected_domain}")

                expanded = self.judge.generate_expanded_queries(ctx.query_text, ctx.topic)
                full_query = f"{ctx.query_text} {ctx.topic} " + " ".join(expanded)
                print(f"   ✓ Sorgu: {len(full_query)} karakter")

                candidates = self.search_engine.retrieve_raw_candidates(full_query)
                if not candidates: continue

                # Mevcut fonksiyonlara ctx içinden okuyarak gönderiyoruz (Geri uyumluluk)
                valid_docs = self.judge.evaluate_candidates(candidates, ctx.query_text, ctx.topic, ctx.negative_scope)
                if not valid_docs: print("🔴 Yargıç hepsini eledi."); continue

                # [V128 EKLENTİSİ] PDF Katmanı için Veri Hazırlığı
                # Ana motor etkilenmez, sadece PDF'e gidecek 'precedent_cards' hazırlanır.
                precedent_cards = self.judge.explain_precedents_for_pdf(valid_docs, ctx.topic)

                context_str = ""
                doc_scan_log = []
                for i, d in enumerate(valid_docs):
                    doc_scan_log.append({
                        "source": d['source'], "page": d['page'],
                        "role": d['role'], "reason": d['reason']
                    })

                    # V122 GÜNCELLEME: Emsal ve Mevzuat Ayrımı
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

                current_personas = {}
                mem_principles = []
                if self.memory_manager:
                    # 1. Fonksiyonu çalıştır (String döner, bunu değişkene atamana gerek yok)
                    self.memory_manager.recall_principles(full_query)

                    # 2. Veriyi 'latest_ui_data' içinden çek (Sözlük buradadır)
                    ui_data = self.memory_manager.latest_ui_data

                    # 3. Şimdi .get() kullanabilirsin
                    if ui_data and ui_data.get("principles"):
                        mem_principles = ui_data["principles"]

                        p_data = mem_principles[0]
                        if "personas_v120" in p_data:
                            current_personas = p_data["personas_v120"]
                        else:
                            current_personas = p_data.get("personas", [])

                # 🔨 Commit 5.4: Decision Context Entegrasyonu
                # Arama ve hafıza sonuçlarını ortak bir yargısal zeminde birleştiriyoruz.
                decision_context = DecisionBuilder.build_decision_context_from_valid_docs(valid_docs)
                decision_context = DecisionBuilder.enrich_decision_context_with_memory(decision_context, mem_principles)

                if not decision_context.has_minimum_legal_basis():
                    print("🔴 KRİTİK UYARI: Yeterli hukuki belge veya ilke bulunamadı. Analiz durduruluyor.")
                    continue

                # 🔨 Commit 5.5: Judge Core (Deterministik Akıl)
                # LLM'e gitmeden önce dosyanın gücünü matematiksel olarak ölçüyoruz.

                judge_core_instance = JudgeCore()
                reflex = judge_core_instance.evaluate(decision_context)

                print(f"   ⚖️  ÖN YARGIÇ REFLEKSİ: {reflex.tendency} (Skor: {reflex.score})")

                if reflex.score < 30:
                    print(f"🔴 Dosya hukuki olarak çok zayıf (Skor: {reflex.score}). Analiz durduruluyor.")
                    continue

                # 🔨 Commit 5.6: Persona Engine (Kontrollü LLM)
                # Hakim tereddütlerine cevap veren yeni persona motoru
                llm_for_persona = ChatOllama(model=LegalConfig.LLM_MODEL, temperature=0.7)  # Biraz daha yaratıcı
                persona_engine = PersonaEngine(llm_for_persona)

                persona_outputs = persona_engine.run(ctx, decision_context, reflex)

                # =========================================================
                # 🧠 ADIM 1-8: MANTIK MOTORU VE SKOR DÜZELTME
                # =========================================================
                # LLM'in ürettiği metinlere bakarak kararı matematiksel olarak güncelle
                # reflex nesnesini EZİYORUZ (Overwrite)
                reflex = self.logic_engine.run_logic(
                    initial_reflex=reflex,
                    persona_outputs=persona_outputs
                )
                # PDF Raporu için persona verilerini güncelle
                # (Eski hafıza verilerini ezerek güncel duruma göre cevap veriyoruz)
                current_personas = {
                    "judge_reflex": reflex.tendency,
                    "doubts": reflex.doubts,
                    "plaintiff": next((p.response for p in persona_outputs if "DAVACI" in p.role), "Beyan yok"),
                    "defendant": next((p.response for p in persona_outputs if "DAVALI" in p.role), "Beyan yok"),
                    "expert": next((p.response for p in persona_outputs if "BİLİRKİŞİ" in p.role), "Beyan yok")
                }

                # 🔨 Commit 5.7: Action Engine (Somut Güçlendirme)
                action_engine = ActionEngine(llm_for_persona)  # Reuse LLM
                strengthening_actions = action_engine.run(reflex, persona_outputs)

                # Avukat Masası (Konsol Çıktısı)
                if strengthening_actions:
                    print(f"\n   🛠️  GÜÇLENDİRME AKSİYONLARI ({len(strengthening_actions)} Adet):")
                    for act in strengthening_actions:
                        print(f"      🔹 [{act.impact_score}/10] {act.title}: {act.description[:100]}...")

                full_advice = self.judge.generate_final_opinion(ctx.query_text, ctx.topic, context_str,judge_reflex=reflex)

                # =========================================================
                # 🚀 COMMIT 6.0 ENTEGRASYONU: RAPOR ORKESTRASYONU
                # =========================================================

                print("\n🖨️  Raporlama Süreci Başlatılıyor...")

                # 1. Orkestratörü Hazırla
                # İsterseniz buraya ClientSummaryPDF() de ekleyebilirsiniz listeye.
                report_orchestrator = ReportOrchestrator(
                    reporters=[
                        LegacyPDFReport(),  # pdf_reports.py içindeki basit legacy
                        JudicialPDFReport()  # pdf_reports.py içindeki gelişmiş judicial
                    ]
                )

                # 2. Tüm Raporları Tek Seferde Üret
                # Not: decision_context (d_ctx) içinden documents listesini çekiyoruz.
                pdf_paths = report_orchestrator.generate_all(
                    context=ctx,  # QueryContext
                    judge_reflex=reflex,  # JudgeReflex (Commit 5.5)
                    persona_outputs=persona_outputs,  # List[PersonaResponse] (Commit 5.6)
                    actions=strengthening_actions,  # List[StrengtheningAction] (Commit 5.7)
                    documents=decision_context.documents  # DecisionContext (Commit 5.4)
                )

                # 3. Sonuçları Bildir
                for path in pdf_paths:
                    print(f"   ✅ Rapor Üretildi: {path}")

                # 4. Müşteri Özeti (Opsiyonel - Veri varsa)
                # Not: client_summary objesi şu an kodda üretilmiyor,
                # eğer üretirseniz burayı açabilirsiniz.
                """
                client_pdf = ClientSummaryPDF()
                client_pdf.generate(client_summary=client_summary_objesi)
                """

                # 5. Konsol Tablosu (Commit 5.2)
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
    freeze_support()
    app = LegalApp()
    app.run()