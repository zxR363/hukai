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
# 3️⃣ ACTIONABLE RECOMMENDATION ENGINE (NEW V95)
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

    def generate(self, judge_concerns):
        """
        Hakim tereddütlerinden somut aksiyon planı üretir.
        judge_concerns: list[str] -> Hakim/Red gerekçesinden çıkarılan maddeler
        """
        recommendations = []

        for concern in judge_concerns:
            # 1. Kategori Belirle (Deterministic)
            category = self._classify_concern(concern)
            if not category:
                # Fallback
                category = "DELIL"

            profile = self.RECOMMENDATION_PROFILE.get(category, self.RECOMMENDATION_PROFILE["DELIL"])

            # 2. İçerik Üret (LLM)
            rec_text = self._generate_recommendation_text(concern, self._category_to_turkish(category))

            # 3. Skor Tahmini (Simulation)
            score_boost = random.randint(profile["base_score_range"][0], profile["base_score_range"][1])

            recommendations.append({
                "recommendation": rec_text,
                "category": category,
                "focus": category,  # V93 uyumluluğu için
                "evidence_type": self._pick_evidence(profile["evidence_type"]),
                "priority": profile["priority"],
                "estimated_cost": profile["estimated_cost"],
                "time_impact": profile["time_impact"],
                "expected_score_increase": score_boost,
                "suggestion": rec_text,  # V93 uyumluluğu için
                "why": concern
            })

        return recommendations

    def _classify_concern(self, concern_text):
        """Kural tabanlı sınıflandırma."""
        text = concern_text.lower()
        if any(k in text for k in ["delil", "ispat", "kanıt", "tanık", "belge", "tespit", "bilirkişi", "rapor"]):
            return "DELIL"
        if any(k in text for k in ["içtihat", "emsal", "yerleşik", "karar", "yargıtay", "daire"]):
            return "ICTIHAT"
        if any(k in text for k in ["usul", "süre", "ehliyet", "şekil", "görev", "yetki", "husumet"]):
            return "USUL"
        if any(k in text for k in ["talep", "fazla", "aşan", "kısmi", "daraltma"]):
            return "TALEP_DARALTMA"
        return None

    def _category_to_turkish(self, category):
        return {
            "DELIL": "delil ve ispat",
            "ICTIHAT": "emsal içtihat",
            "USUL": "usul hukuku",
            "TALEP_DARALTMA": "stratejik talep"
        }.get(category, "hukuki")

    def _generate_recommendation_text(self, concern, category_tr):
        prompt = f"""
Bir avukata yol gösterecek şekilde, aşağıdaki hakim tereddüdüne yönelik {category_tr} odaklı SOMUT bir aksiyon önerisi yaz.

Hakim Tereddüdü: "{concern}"

Kurallar:
- Tek bir cümle yaz.
- Emir kipi kullan (Örn: "... sunulmalıdır", "... yapılmalıdır").
- Hukuki ve profesyonel olsun.

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
# 4️⃣ HAFIZA YÖNETİCİSİ (FULL INTEGRATED)
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
        # V95 Entegrasyonu
        self.recommendation_engine = ActionableRecommendationEngine(llm)

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
        prompt = f"""
GÖREV: Aşağıdaki hukuki ilkenin yönünü belirle.
İLKE: "{principle_text}"
CEVAP (SADECE BİRİ): [LEHINE] veya [ALEYHINE] veya [BELIRSIZ]
"""
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

    def _calculate_case_success_probability(self, principle_confidence, trend_direction, conflict, domain_match):
        score = principle_confidence * 100
        if trend_direction == "up":
            score += 10
        elif trend_direction == "down":
            score -= 10
        if conflict: score -= 15
        if not domain_match: score -= 10

        score = max(0, min(100, round(score, 1)))

        conf_level = "Yüksek" if score >= 70 else "Orta" if score >= 40 else "Düşük"
        summary = "Başarı ihtimali yüksek." if score >= 70 else "Başarı ihtimali orta, riskli." if score >= 40 else "Başarı ihtimali düşük."

        return {
            "success_probability": score,
            "confidence_level": conf_level,
            "summary": summary
        }

    def _derive_persona_signals(self, analysis_data, item_data):
        judge_score = analysis_data['success_probability']
        judge = {
            "stance": "strong" if judge_score > 70 or judge_score < 30 else "weak",
            "direction": "acceptance" if judge_score >= 50 else "rejection",
            "confidence_level": "high" if judge_score > 80 else "medium"
        }
        prosecutor_dir = "rejection" if (item_data['conflict'] or item_data['trend_dir'] == 'down') else "acceptance"
        prosecutor = {
            "stance": "strong",
            "direction": prosecutor_dir,
            "confidence_level": "high"
        }
        expert = {
            "stance": "neutral",
            "direction": "cautious",
            "confidence_level": "medium"
        }
        return {"judge": judge, "prosecutor": prosecutor, "expert": expert}

    def _analyze_persona_conflict(self, personas):
        score = 0
        reasons = []
        if personas["prosecutor"]["direction"] != personas["judge"]["direction"]:
            score += 40
            reasons.append("Yargısal yönler zıt")
        if personas["prosecutor"]["stance"] == "strong" and personas["judge"]["stance"] == "weak":
            score += 30
            reasons.append("Savcı güçlü, hakim ihtiyatlı")

        return {
            "conflict_score": min(score, 100),
            "conflict_level": "Yüksek" if score >= 70 else "Orta" if score >= 40 else "Düşük",
            "summary": reasons
        }

    def _simulate_net_decision(self, personas):
        dir_map = {"acceptance": 1, "cautious": 0, "rejection": -1}
        stance_map = {"strong": 1.0, "neutral": 0.6, "weak": 0.3}
        conf_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
        weights = {"judge": 0.60, "prosecutor": 0.25, "expert": 0.15}

        total = 0
        breakdown = {}
        for name, data in personas.items():
            s = dir_map.get(data["direction"], 0) * stance_map.get(data["stance"], 0.6) * conf_map.get(
                data["confidence_level"], 0.7) * weights.get(name, 0)
            breakdown[name] = round(s, 3)
            total += s

        decision = "KABUL EĞİLİMLİ" if total >= 0.25 else "RED EĞİLİMLİ" if total <= -0.25 else "Belirsiz / Riskli"
        return {"final_score": round(total, 3), "decision": decision, "breakdown": breakdown}

    # --- GENERATORS ---
    def _generate_judicial_reasoning(self, analysis):
        prompt = f"SEN TÜRK HAKİMİSİN. Verilen veriyi ({analysis['success_probability']} skor) yargısal dille özetle. Kısa."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_prosecutor_reasoning(self, analysis):
        prompt = f"SEN SAVCISIN. Verilen veriyi ({analysis['success_probability']} skor) iddia makamı diliyle özetle. Kısa."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_expert_witness_reasoning(self, analysis):
        prompt = f"SEN BİLİRKİŞİSİN. Verilen veriyi ({analysis['success_probability']} skor) teknik dille özetle. Kısa."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_rejection_reasoning(self, analysis):
        prompt = f"SEN HAKİMSİN. Bu davayı REDDETSEYDİN gerekçen ne olurdu? ({analysis['success_probability']} skor). Kısa."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_final_verdict_reasoning(self, net_decision, topic, trend, principles):
        prompt = f"SEN HAKİMSİN. {net_decision['decision']} yönündeki kararın gerekçesini yaz. Konu: {topic}. Kısa."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    def _generate_executive_summary(self, net_decision, judge, pros, exp, trend):
        prompt = f"SEN YÖNETİCİSİN. Dava risk özetini tek paragrafta yaz. Karar: {net_decision['decision']}."
        try:
            return self.llm.invoke(prompt).content.strip()
        except:
            return ""

    # --- V94: EXTRACT CONCERNS ---
    def _extract_concerns_for_engine(self, text):
        prompt = f"Aşağıdaki metindeki temel hukuki zayıflıkları veya riskleri 3 kısa madde halinde listele.\nMETİN:\n{text}"
        try:
            res = self.llm.invoke(prompt).content.strip()
            return [line.strip("- *") for line in res.splitlines() if len(line) > 5][:3]
        except:
            return ["Genel ispat eksikliği", "İçtihat belirsizliği"]

    # --- V93: IMPACT ESTIMATION ---
    def _estimate_mitigation_impact(self, rec_text, min_val, max_val):
        prompt = f"Aşağıdaki önerinin dava başarısına etkisini ({min_val}-{max_val}) arası bir rakamla puanla. Sadece rakam yaz.\nÖNERİ: {rec_text}"
        try:
            res = self.llm.invoke(prompt).content.strip()
            val = int(re.findall(r"\d+", res)[0])
            return max(min(val, max_val), min_val)
        except:
            return min_val

    def _simulate_post_strengthening_score(self, base_score, recommendations):
        total_boost = 0
        drivers = []
        for rec in recommendations:
            cat = rec.get("category", "DELIL")
            cfg = self.MITIGATION_EFFECTS.get(cat, {"min": 1, "max": 3})
            impact = self._estimate_mitigation_impact(rec["recommendation"], cfg["min"], cfg["max"])
            total_boost += impact
            drivers.append(f"{cat}(+{impact})")

        total_boost = min(total_boost, self.MAX_TOTAL_BOOST)
        new_score = min(base_score + total_boost, self.MAX_SCORE)
        return {"current_score": base_score, "projected_score": new_score, "total_boost": total_boost,
                "drivers": drivers}

    # --- MAIN RECALL FUNCTION ---
    def recall_principles(self, query_text):
        try:
            query_domain = self._detect_domain_from_query(query_text)
            vector = self.embedder.embed_query(query_text)
            hits = self.client.query_points(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                query=vector, limit=15
            ).points

            processed_hits = []
            for h in hits:
                raw_conf = h.payload.get("confidence", 0.5)
                ts = h.payload.get("timestamp", time.time())
                domain = h.payload.get("domain", "Genel")
                evolution_note = h.payload.get("evolution_note", "")
                final_conf = self._apply_time_decay(raw_conf, ts)
                is_domain_match = (query_domain.lower() in domain.lower())

                if final_conf >= LegalConfig.MIN_CONFIDENCE_THRESHOLD:
                    trend_dir = "up" if "GÜÇLENEN" in evolution_note else "down" if "ZAYIFLAYAN" in evolution_note else "stable"
                    item = {
                        "text": h.payload['principle'], "conf": final_conf, "domain": domain,
                        "conflict": h.payload.get("conflict_flag", False), "score": h.score,
                        "trend_dir": trend_dir, "domain_match": is_domain_match, "evolution_note": evolution_note
                    }
                    processed_hits.append(item)

            sorted_hits = sorted(processed_hits, key=lambda x: x["score"], reverse=True)[:3]
            if not sorted_hits: return ""

            memory_text = f"\n💡 YERLEŞİK İÇTİHAT HAFIZASI ({query_domain} Alanı):\n"

            for item in sorted_hits:
                # 1. Analizler
                analysis = self._calculate_case_success_probability(
                    item["conf"], item["trend_dir"], item["conflict"], item["domain_match"]
                )
                persona_signals = self._derive_persona_signals(analysis, item)
                net_decision = self._simulate_net_decision(persona_signals)

                # 2. Metin Üretimi
                judicial_text = self._generate_judicial_reasoning(analysis)
                prosecutor_text = self._generate_prosecutor_reasoning(analysis)
                expert_text = self._generate_expert_witness_reasoning(analysis)
                rejection_text = self._generate_rejection_reasoning(analysis)
                verdict_text = self._generate_final_verdict_reasoning(net_decision, query_text, item['evolution_note'],
                                                                      item['text'])
                exec_summary = self._generate_executive_summary(net_decision, judicial_text, prosecutor_text,
                                                                expert_text, item['evolution_note'])

                # 3. Aksiyon Planı ve Simülasyon
                concerns = self._extract_concerns_for_engine(judicial_text + "\n" + rejection_text)
                action_plan = self.recommendation_engine.generate(concerns)
                simulation_result = self._simulate_post_strengthening_score(analysis['success_probability'],
                                                                            action_plan)

                if self.last_recalled_query != query_text:
                    print("\n" + "=" * 70)
                    print(f"📊 [AKILLI STRATEJİ VE AKSİYON PLANI] (V95: Grand Unified System)")
                    print(f"   🎯 KONU: {query_text} | ⚖️ DURUM: {net_decision['decision']}")
                    print("-" * 70)
                    print(f"📝 YÖNETİCİ ÖZETİ: \"{exec_summary[:120]}...\"")
                    print("-" * 70)
                    print("🚀 SOMUT AKSİYON PLANI (REÇETE):")
                    for act in action_plan:
                        print(f"   🔧 [{act['category']}] {act['recommendation']}")
                        print(
                            f"      ↳ 💰 Maliyet: {act['estimated_cost']} | ⏳ Zaman: {act['time_impact']} | 📈 Etki: +{act['expected_score_increase']}")
                    print("-" * 70)
                    print(
                        f"🔮 SİMÜLASYON: %{analysis['success_probability']} --> %{simulation_result['projected_score']} (Potansiyel Başarı)")
                    print("=" * 70 + "\n")

                warning = "⚠️ [YARGISAL ÇELİŞKİ]" if item["conflict"] else ""
                memory_text += f"- {warning} [{item['domain']}] {item['text']}\n"
                memory_text += f"  📝 ÖZET: {exec_summary}\n"
                memory_text += f"  🏆 DURUM: {net_decision['decision']} (%{analysis['success_probability']})\n"
                memory_text += f"  ✍️ KARAR TASLAĞI: {verdict_text}\n"

                memory_text += "\n  🚀 GÜÇLENDİRME PLANI:\n"
                for act in action_plan:
                    memory_text += f"  • {act['recommendation']} (+{act['expected_score_increase']} Puan)\n"

            self.last_recalled_query = query_text
            return memory_text
        except Exception as e:
            print(f"Hata: {e}")
            return ""

    def calculate_memory_consensus(self, source_name, current_decision, vector_score):
        try:
            scroll_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))])
            points, _ = self.client.scroll(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["decision"],
                scroll_filter=scroll_filter,
                limit=20
            )

            if not points:
                if vector_score > 0.80: return 1.10
                return 1.0

            match_count = sum(1 for p in points if p.payload.get("decision") == current_decision)
            total = len(points)
            ratio = match_count / total

            if ratio > 0.8: return 1.15
            if ratio < 0.2: return 0.85
            return 1.0
        except:
            return 1.0

    def save_decision(self, query, doc_name, decision, reason, doc_type):
        try:
            text_to_embed = f"{query} {doc_name} {decision} {reason}"
            vector = self.embedder.embed_query(text_to_embed)

            payload = {
                "query": query,
                "source": doc_name,
                "decision": decision,
                "reason": reason,
                "doc_type": doc_type,
                "timestamp": time.time(),
                "created_at": datetime.now().isoformat(),
                "id": str(uuid.uuid4())
            }

            self.client.upsert(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["decision"],
                points=[PointStruct(id=payload['id'], vector=vector, payload=payload)]
            )
        except Exception as e:
            print(f"⚠️ Hafıza hatası: {e}")

    # --- MATEMATİKSEL YARDIMCILAR ---
    def _cosine_similarity(self, v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0: return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def _calculate_vector_mean(self, vectors):
        if not vectors: return []
        dim = len(vectors[0])
        count = len(vectors)
        mean_vec = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                mean_vec[i] += vec[i]
        return [x / count for x in mean_vec]

    def _cluster_reasonings(self, items, threshold=0.86):
        clusters = []
        for item in items:
            added = False
            for cluster in clusters:
                sim = self._cosine_similarity(item['vector'], cluster['centroid'])
                if sim >= threshold:
                    cluster['members'].append(item)
                    all_vectors = [m['vector'] for m in cluster['members']]
                    cluster['centroid'] = self._calculate_vector_mean(all_vectors)
                    added = True
                    break
            if not added:
                clusters.append({'members': [item], 'centroid': item['vector']})
        return [c['members'] for c in clusters]

    def _calculate_principle_confidence(self, cluster):
        count = len(cluster)
        count_score = min(1.0, count / 10)
        if count > 1:
            vectors = [c['vector'] for c in cluster]
            centroid = self._calculate_vector_mean(vectors)
            sims = [self._cosine_similarity(v, centroid) for v in vectors]
            similarity_score = sum(sims) / len(sims)
        else:
            similarity_score = 1.0
        return round((count_score * 0.6) + (similarity_score * 0.4), 2)

    def _analyze_trend_momentum(self, trend_dict):
        buckets = sorted(trend_dict.keys())
        if not buckets: return "Veri Yok"
        oldest = buckets[0]
        newest = buckets[-1]

        def get_rate(bucket_key):
            d = trend_dict[bucket_key]
            total = d["KABUL"] + d["RED"]
            if total == 0: return 0.0
            return d["KABUL"] / total

        old_rate = get_rate(oldest)
        new_rate = get_rate(newest)
        delta = new_rate - old_rate
        if newest == "2022-2024" and (trend_dict[newest]["KABUL"] + trend_dict[newest]["RED"] == 0):
            return "🕸️ ESKİ İÇTİHAT: Güncel (2022+) bir teyit bulunamadı."
        if delta > 0.2:
            return f"📈 GÜÇLENEN İÇTİHAT: {oldest} %{old_rate * 100:.0f} -> %{new_rate * 100:.0f}."
        elif delta < -0.2:
            return f"📉 ZAYIFLAYAN EĞİLİM: {oldest} %{old_rate * 100:.0f} -> %{new_rate * 100:.0f}."
        elif new_rate > 0.8:
            return "💎 YERLEŞİK VE GÜNCEL."
        elif new_rate < 0.3:
            return "❌ TERK EDİLMİŞ GÖRÜŞ."
        else:
            return "⚖️ DALGALI SEYİR."

    def _save_principle_v79(self, text, confidence, source_ids, domain, cluster_data):
        try:
            vec = self.embedder.embed_query(text)
            polarity = self._detect_polarity(text)

            hits = self.client.query_points(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                query=vec,
                limit=10,
                score_threshold=0.80
            ).points

            conflict = False
            trend = Counter()
            polarity_stats = {"LEHINE": 0, "ALEYHINE": 0, "BELIRSIZ": 0}

            if polarity in polarity_stats: polarity_stats[polarity] += 1
            for h in hits:
                p = h.payload.get("polarity", "BELIRSIZ")
                if p in polarity_stats: polarity_stats[p] += 1
                if (p == "LEHINE" and polarity == "ALEYHINE") or (p == "ALEYHINE" and polarity == "LEHINE"):
                    conflict = True

            total_hits = sum(polarity_stats.values())
            lehine_pct = (polarity_stats["LEHINE"] / total_hits * 100) if total_hits > 0 else 0
            aleyhine_pct = (polarity_stats["ALEYHINE"] / total_hits * 100) if total_hits > 0 else 0

            yearly_stats = {}
            for c in cluster_data:
                ts = c.get("timestamp", time.time())
                year = datetime.fromtimestamp(ts).year
                decision = c.get("decision", "KABUL")
                source = c.get("source", "Bilinmeyen Dosya")

                bucket = self._extract_year_bucket(ts)
                trend[(bucket, decision)] += 1

                if year not in yearly_stats: yearly_stats[year] = {"KABUL": [], "RED": []}
                if decision in yearly_stats[year]: yearly_stats[year][decision].append(source)

            trend_dict = {}
            for (bucket, dec), count in trend.items():
                if bucket not in trend_dict: trend_dict[bucket] = {"KABUL": 0, "RED": 0}
                trend_dict[bucket][dec] = count

            evolution_msg = self._analyze_trend_momentum(trend_dict)

            print("\n" + "=" * 60)
            print(f"📊 [ANALİZ LOGU] İÇTİHAT EVRİM RAPORU (V95)")
            print("-" * 60)
            print(f"🔹 İLKE: '{text[:80]}...'")
            print(f"🔹 ALAN: {domain}")
            print(f"🔹 YÖN: {polarity}")
            print(f"🔹 ÇELİŞKİ: {'⚠️ VAR' if conflict else '✅ YOK'}")
            print(f"🧠 İÇTİHAT EVRİMİ: {evolution_msg}")
            print("-" * 30)
            print(f"🔹 KAYNAK DOSYA VE YILLIK KIRILIM:")
            for yr in sorted(yearly_stats.keys()):
                d = yearly_stats[yr]
                print(f"   📅 {yr}:")
                if d['KABUL']:
                    print(f"      ✅ LEHINE ({len(d['KABUL'])}):")
                    for s in d['KABUL']: print(f"         - {s}")
                if d['RED']:
                    print(f"      ❌ ALEYHINE ({len(d['RED'])}):")
                    for s in d['RED']: print(f"         - {s}")
            print("=" * 60 + "\n")

            payload = {
                "principle": text,
                "confidence": confidence,
                "domain": domain,
                "polarity": polarity,
                "trend": trend_dict,
                "conflict_flag": conflict,
                "source_count": len(source_ids),
                "source_ids": source_ids,
                "evolution_note": evolution_msg,
                "generated_by": "consolidation_v95",
                "timestamp": time.time(),
                "created_at": datetime.now().isoformat()
            }

            self.client.upsert(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                points=[PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)]
            )

        except Exception as e:
            print(f"⚠️ İlke kaydetme hatası: {e}")

    def consolidate_principles_v79(self):
        print("\n🔥 İÇTİHAT MİMARI: Artımlı Konsolidasyon (V95: Full Power)...")
        try:
            time_filter = Filter(
                must=[
                    FieldCondition(
                        key="timestamp",
                        range=Range(gt=self.last_consolidation_ts)
                    )
                ]
            )

            points, _ = self.client.scroll(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["decision"],
                scroll_filter=time_filter,
                limit=200
            )

            candidates = []
            for p in points:
                if (p.payload.get('doc_type') == 'EMSAL KARAR' and
                        len(p.payload.get('reason', '')) > 30):

                    if p.payload.get('decision') == 'KABUL':
                        candidates.append({
                            "reason": p.payload['reason'],
                            "id": p.id,
                            "source": p.payload.get('source', 'Bilinmeyen'),
                            "timestamp": p.payload.get('timestamp', time.time()),
                            "decision": p.payload.get('decision'),
                            "vector": None
                        })

            if len(candidates) < 3:
                print("   ℹ️ Son konsolidasyondan beri yeterli yeni veri birikmedi.")
                return

            print(f"   🔍 {len(candidates)} adet YENİ gerekçe analiz ediliyor...")

            texts = [c["reason"] for c in candidates]
            vectors = self.embedder.embed_documents(texts)
            for i, vec in enumerate(vectors):
                candidates[i]["vector"] = vec

            clusters = self._cluster_reasonings(candidates, threshold=0.86)
            print(f"   🧩 {len(clusters)} farklı hukuki desen bulundu.")

            for i, cluster in enumerate(clusters):
                if len(cluster) < 3: continue

                print(f"   ⚙️ Küme #{i + 1} işleniyor...")
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
                    domain_text = domain_match.group(1) if domain_match else "Genel Hukuk"

                    conf = self._calculate_principle_confidence(cluster)
                    source_ids = [c['id'] for c in cluster]

                    self._save_principle_v79(principle_text, conf, source_ids, domain_text, cluster)

            self._save_state()
            print("✅ Konsolidasyon tamamlandı ve durum kaydedildi.")

        except Exception as e:
            print(f"⚠️ Konsolidasyon hatası: {e}")


# ==================================================
# 4️⃣ ARAMA MOTORU SINIFI (SEARCH ENGINE)
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

        candidates = sorted(unique_docs.values(), key=lambda x: x.score, reverse=True)[:self.config.LLM_RERANK_LIMIT]

        if not candidates: print("🔴 Uygun belge bulunamadı."); return []
        print(f"   ✅ {len(candidates)} potansiyel belge bulundu. Yargıca gönderiliyor...")
        return candidates


# ==================================================
# 5️⃣ YARGIÇ VE MUHAKEME SINIFI (JUDGE)
# ==================================================
class LegalJudge:
    def __init__(self, memory_manager=None):
        self.llm = ChatOllama(model=LegalConfig.LLM_MODEL, temperature=0.1)
        self.memory = memory_manager

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

    def _check_relevance_judge_smart(self, user_query, user_filter, negative_keywords, document_text, source_name):
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

        prompt_gen = f"""
SEN KIDEMLI BIR HUKUKCUSSUN.
{memory_context}

SORGUNUN AMACI: Benzer Yargıtay içtihatlarını bulmak.
Sorgu: "{user_query}"
Belge: "{document_text[:700]}..."
SORU: Bu belge; hukuki ilke, yorum yaklaşımı, miras hukuku mantığı bakımından sorguyla ne derece BENZER?
SADECE BİRİNİ SEÇ: [ÇOK BENZER], [BENZER], [ZAYIF]
Altına tek cümlelik gerekçe yaz.
"""
        res = self.llm.invoke(prompt_gen).content.strip()
        is_ok = ("ÇOK BENZER" in res) or ("BENZER" in res)
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
        print("\n⚖️  Akıllı Yargıç Değerlendiriyor (V95: Full Power):")
        valid_docs = []

        for hit in candidates:
            doc_text = hit.payload['page_content']
            source = hit.payload['source']
            page = hit.payload['page']
            type_desc = hit.payload['type']

            is_ok, reason = self._check_relevance_judge_smart(story, topic, negatives, doc_text, source)

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
                elif consensus_multiplier == 1.10:
                    log_score += " (✨ KEŞİF BONUSU)"
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

    def generate_final_opinion(self, story, topic, context_str):
        print("\n🧑‍⚖️  AVUKAT YAZIYOR (V95: Full Analysis)...")

        system_content = """SEN KIDEMLİ BİR HUKUKÇUSUN.
GÖREVİN: Sana verilen "DELİLLER" listesindeki Yargıç notlarını derleyerek nihai raporu yazmak.

KURALLAR:
1. SADECE Yargıç'ın "Gerekçe" veya "Sebep" olarak yazdığı bilgileri temel al.
2. Belgelerin içindeki konuyla alakasız (harç iadesi, usul detayları vb.) kısımları GÖRMEZDEN GEL.
3. ASLA aynı bilgiyi tekrar etme.
4. Çıktıyı tam olarak şu başlıklarla ver:

A. MEVZUAT DAYANAKLARI
(Burada sadece MEVZUAT etiketli belgeleri özetle)

B. İLGİLİ EMSAL KARARLAR
(Burada EMSAL KARAR etiketli belgeleri, Yargıç'ın belirlediği ROL'e göre, Yargıç Gerekçesi'ni kullanarak anlat)

C. SONUÇ VE HUKUKİ TAVSİYE
(Kullanıcının olayına göre, bulunan emsallere dayanarak net bir yol haritası çiz)"""

        user_content = f"""Aşağıdaki "DELİLLER" listesinde sunulan belgeleri kullanarak olayı analiz et.
OLAY: "{story}"
ODAK: "{topic}"
DELİLLER:
{context_str}
ANALİZİ BAŞLAT:"""

        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]

        full_res = ""
        for chunk in self.llm.stream(messages):
            c = chunk.content;
            full_res += c;
            print(c, end="", flush=True)
        print("\n")
        return full_res


# ==================================================
# 6️⃣ RAPORLAMA SINIFI (REPORTER)
# ==================================================
class PDFReportGenerator(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'HUKUKI ANALIZ RAPORU', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C');
        self.ln(5)

    def footer(self):
        self.set_y(-15);
        self.set_font('helvetica', 'I', 8);
        self.cell(0, 10, f'Sayfa {self.page_no()}', align='C')


class LegalReporter:
    @staticmethod
    def create_report(user_story, valid_docs, advice_text, filename="Hukuki_Rapor_V95.pdf"):
        pdf = PDFReportGenerator();
        pdf.add_page();
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
            pdf.set_font(style='B', size=9)
            source_title = f"[{doc['type']}] {doc['source']} (Sf. {doc['page']})"
            pdf.cell(0, 6, clean(source_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(style='B', size=8)
            pdf.cell(0, 5, clean(f"   Rol: {doc['role']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font(style='I', size=8);
            pdf.multi_cell(w=pdf.epw, h=4, text=clean(f"   Sebep: {doc['reason']}"));
            pdf.ln(2)

        pdf.add_page();
        pdf.set_font(style='B', size=12);
        pdf.cell(0, 10, clean("3. HUKUKI GORUS:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(style='', size=10);
        pdf.multi_cell(w=pdf.epw, h=6, text=clean(advice_text))
        try:
            pdf.output(filename);
            print(f"\n📄 Rapor Hazır: {filename}")
        except:
            pass


# ==================================================
# 7️⃣ ANA UYGULAMA (MAIN APP)
# ==================================================
class LegalApp:
    def __init__(self):
        print("🚀 LEGAL SUITE V95 (Grand Unified System: All Engines Go!)...")
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
                for i, d in enumerate(valid_docs):
                    context_str += f"""
                        BELGE #{i + 1}
                        KAYNAK: {d['source']}
                        TÜR: {d['type']}
                        ROL: {d['role']}
                        YARGIÇ GEREKÇESİ: {d['reason']}
                        İÇERİK ÖZETİ: {d['text'][:800]}...
                        =========================================
                        """
                print("\n" + "=" * 30)
                print("### Kaynaklar ve Sebebi")
                print("=" * 30)
                for d in valid_docs:
                    print(f"• [{d['type']}] {d['source']} (Sf. {d['page']}) | Skor: %{d['score']:.1f}")
                    print(f"  Rol:   {d['role']}")
                    print(f"  Sebep: {d['reason']}")
                    print("-" * 40)

                full_advice = self.judge.generate_final_opinion(story, topic, context_str)
                self.reporter.create_report(story, valid_docs, full_advice)

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