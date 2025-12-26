import sys
import os
import re
import uuid
import time
import shutil
import atexit
import json
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
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
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
        "decision": "judge_memory_v1",  # Karar Hafızası
        "principle": "principle_memory_v1"  # İlke Hafızası
    }

    QDRANT_PATH = "qdrant_db_master"
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "qwen2.5"
    SEARCH_LIMIT_PER_SOURCE = 30
    SCORE_THRESHOLD = 0.40
    LLM_RERANK_LIMIT = 5

    # V72 Ayarları
    DECAY_RATE_PER_MONTH = 0.98  # Her ay güven %2 azalır
    PRINCIPLE_MERGE_THRESHOLD = 0.90  # %90 benzerse yeni kayıt açma, birleştir


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
# 3️⃣ HAFIZA YÖNETİCİSİ (V72: ARCHITECT EDITION)
# ==================================================
class LegalMemoryManager:
    def __init__(self, client, embedder, llm):
        self.client = client
        self.embedder = embedder
        self.llm = llm
        self._init_memory_collections()

    def _init_memory_collections(self):
        for name, col_name in LegalConfig.MEMORY_COLLECTIONS.items():
            if not self.client.collection_exists(col_name):
                print(f"🧠 Hafıza oluşturuluyor: {col_name}")
                self.client.create_collection(col_name, vectors_config=VectorParams(size=768, distance=Distance.COSINE))

    # --- YENİ: ZAMAN AŞIMI HESABI (TIME DECAY) ---
    def _apply_time_decay(self, confidence, timestamp):
        """
        Eski bilgilerin güvenini azaltır.
        Formül: Confidence * (0.98 ^ Ay_Sayisi)
        """
        if not timestamp: return confidence

        elapsed_seconds = time.time() - timestamp
        elapsed_months = elapsed_seconds / (30 * 24 * 3600)  # Yaklaşık ay sayısı

        decay_factor = math.pow(LegalConfig.DECAY_RATE_PER_MONTH, elapsed_months)
        return confidence * decay_factor

    def recall_principles(self, query_text):
        try:
            vector = self.embedder.embed_query(query_text)
            hits = self.client.query_points(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                query=vector,
                limit=10  # Havuzu genişlet, sonra filtrele
            ).points

            # Sonuçları işle: Decay uygula ve Sırala
            processed_hits = []
            for h in hits:
                raw_conf = h.payload.get("confidence", 0.5)
                ts = h.payload.get("timestamp", time.time())

                # 3. İYİLEŞTİRME: Güven Zamanla Düşsün
                final_conf = self._apply_time_decay(raw_conf, ts)

                # Çok zayıflayanları ele (< %20)
                if final_conf > 0.20:
                    processed_hits.append({
                        "text": h.payload['principle'],
                        "conf": final_conf,
                        "domain": h.payload.get("domain", "Genel")
                    })

            # Puana göre sırala ve ilk 3'ü al
            sorted_hits = sorted(processed_hits, key=lambda x: x["conf"], reverse=True)[:3]

            if not sorted_hits: return ""

            memory_text = "\n💡 YERLEŞİK İÇTİHAT HAFIZASI (Sistem Geçmişi):\n"
            for item in sorted_hits:
                memory_text += f"- [{item['domain']}] {item['text']} (Güven: %{item['conf'] * 100:.0f})\n"
            return memory_text
        except:
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
                "id": str(uuid.uuid4())  # ID'yi payload'a da ekle, lazım olabilir
            }

            self.client.upsert(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["decision"],
                points=[PointStruct(id=payload['id'], vector=vector, payload=payload)]
            )
        except Exception as e:
            print(f"⚠️ Hafıza hatası: {e}")

    # --- MATEMATİKSEL YARDIMCILAR (V72) ---
    def _cosine_similarity(self, v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0: return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def _calculate_vector_mean(self, vectors):
        """
        4. İYİLEŞTİRME: Küme Ortalamasını (Centroid) Al
        """
        if not vectors: return []
        dim = len(vectors[0])
        count = len(vectors)
        mean_vec = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                mean_vec[i] += vec[i]
        return [x / count for x in mean_vec]

    def _cluster_reasonings(self, items, threshold=0.86):
        """
        4. İYİLEŞTİRME: Centroid bazlı kümeleme
        """
        clusters = []  # List of {'members': [], 'centroid': []}

        for item in items:
            added = False
            for cluster in clusters:
                # Gerçek centroid ile karşılaştır
                sim = self._cosine_similarity(item['vector'], cluster['centroid'])
                if sim >= threshold:
                    cluster['members'].append(item)
                    # Centroid'i güncelle (Basit hareketli ortalama yerine yeniden hesapla)
                    all_vectors = [m['vector'] for m in cluster['members']]
                    cluster['centroid'] = self._calculate_vector_mean(all_vectors)
                    added = True
                    break

            if not added:
                clusters.append({
                    'members': [item],
                    'centroid': item['vector']
                })

        # Sadece member listelerini döndür
        return [c['members'] for c in clusters]

    def _calculate_principle_confidence(self, cluster):
        count = len(cluster)
        count_score = min(1.0, count / 10)

        if count > 1:
            # Küme merkezine ortalama uzaklık
            vectors = [c['vector'] for c in cluster]
            centroid = self._calculate_vector_mean(vectors)
            sims = [self._cosine_similarity(v, centroid) for v in vectors]
            similarity_score = sum(sims) / len(sims)
        else:
            similarity_score = 1.0

        return round((count_score * 0.6) + (similarity_score * 0.4), 2)

    # --- V72: GELİŞMİŞ İLKE KAYDETME (DEDUPLICATION & SOURCE LINKING) ---
    def _save_principle_v72(self, text, confidence, source_ids, domain):
        """
        1. İYİLEŞTİRME: İlke tekrarlarını engelle (Merge)
        2. İYİLEŞTİRME: Domain etiketi ekle
        5. İYİLEŞTİRME: İlke <-> Karar bağlantısı (source_ids)
        """
        try:
            vec = self.embedder.embed_query(text)

            # Önce benzer ilke var mı diye bak
            hits = self.client.query_points(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                query=vec,
                limit=1,
                score_threshold=LegalConfig.PRINCIPLE_MERGE_THRESHOLD  # %90 Benzerlik
            ).points

            if hits:
                # VARSA GÜNCELLE (MERGE)
                existing = hits[0]
                old_ids = existing.payload.get("source_ids", [])
                new_ids = list(set(old_ids + source_ids))  # ID'leri birleştir

                new_count = len(new_ids)
                new_conf = min(1.0, existing.payload['confidence'] + 0.05)  # Güveni biraz artır

                self.client.set_payload(
                    collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                    points=[existing.id],
                    payload={
                        "confidence": new_conf,
                        "source_count": new_count,
                        "source_ids": new_ids,
                        "timestamp": time.time()  # Taze bilgi, zaman aşımını sıfırla
                    }
                )
                print(f"   ♻️ Mevcut ilke GÜÇLENDİRİLDİ (Yeni Güven: %{new_conf * 100:.0f})")

            else:
                # YOKSA YENİ OLUŞTUR
                self.client.upsert(
                    collection_name=LegalConfig.MEMORY_COLLECTIONS["principle"],
                    points=[PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "principle": text,
                            "confidence": confidence,
                            "source_count": len(source_ids),
                            "source_ids": source_ids,  # Kaynak Bağlantısı
                            "domain": domain,  # Alan Bilgisi
                            "generated_by": "consolidation_v72",
                            "timestamp": time.time()
                        }
                    )]
                )
                print(f"   🏆 YENİ İÇTİHAT OLUŞTURULDU (Güven: %{confidence * 100:.0f}, Alan: {domain})")

        except Exception as e:
            print(f"⚠️ İlke kaydetme hatası: {e}")

    # --- V72: KONSOLİDASYON ANA FONKSİYONU ---
    def consolidate_principles_v72(self):
        print("\n🔥 İÇTİHAT MİMARI: İlke-Temelli Konsolidasyon (V72)...")
        try:
            points, _ = self.client.scroll(
                collection_name=LegalConfig.MEMORY_COLLECTIONS["decision"],
                limit=150  # Daha fazla veri çek
            )

            # Veri Hazırlığı: Sadece Emsaller, ID'leri ile birlikte
            candidates = []
            for p in points:
                if (p.payload.get('doc_type') == 'EMSAL KARAR' and
                        p.payload.get('decision') == 'KABUL' and
                        len(p.payload.get('reason', '')) > 30):
                    candidates.append({
                        "reason": p.payload['reason'],
                        "id": p.id,  # 5. İyileştirme için ID lazım
                        "vector": None  # Sonra eklenecek
                    })

            if len(candidates) < 3:
                print("   ℹ️ Yeterli emsal birikmedi.")
                return

            print(f"   🔍 {len(candidates)} adet gerekçe analiz ediliyor...")

            # Vektörleştirme
            texts = [c["reason"] for c in candidates]
            vectors = self.embedder.embed_documents(texts)
            for i, vec in enumerate(vectors):
                candidates[i]["vector"] = vec

            # Kümeleme (Gerçek Centroid ile)
            clusters = self._cluster_reasonings(candidates, threshold=0.86)
            print(f"   🧩 {len(clusters)} farklı hukuki desen bulundu.")

            for i, cluster in enumerate(clusters):
                if len(cluster) < 3: continue

                print(f"   ⚙️ Küme #{i + 1} işleniyor...")

                # LLM'e gönder
                reasonings_text = "\n".join([f"- {c['reason']}" for c in cluster])

                # 2. İYİLEŞTİRME: Domain Çıkarımı
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

                # Regex ile parse et
                principle_match = re.search(r"İLKE:\s*(.*)", res)
                domain_match = re.search(r"ALAN:\s*(.*)", res)

                if principle_match:
                    principle_text = principle_match.group(1)
                    domain_text = domain_match.group(1) if domain_match else "Genel Hukuk"

                    # Güven Hesabı
                    conf = self._calculate_principle_confidence(cluster)

                    # Kaynak ID'lerini topla
                    source_ids = [c['id'] for c in cluster]

                    # AKILLI KAYDETME (Merge & Link)
                    self._save_principle_v72(principle_text, conf, source_ids, domain_text)

            print("✅ İçtihat mimarisi güncellendi.")

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
                                                   limit=40).points
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
        print("\n⚖️  Akıllı Yargıç Değerlendiriyor (V72: Architect Edition):")
        valid_docs = []

        for hit in candidates:
            doc_text = hit.payload['page_content']
            source = hit.payload['source']
            page = hit.payload['page']
            type_desc = hit.payload['type']

            is_ok, reason = self._check_relevance_judge_smart(story, topic, negatives, doc_text, source)

            # Konsensüs Skoru & Keşif Bonusu
            consensus_multiplier = 1.0
            if self.memory:
                consensus_decision = "KABUL" if is_ok else "RED"
                # hit.score gönderiliyor
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
        print("\n🧑‍⚖️  AVUKAT YAZIYOR (V72: Full Analysis)...")

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
    def create_report(user_story, valid_docs, advice_text, filename="Hukuki_Rapor_V72.pdf"):
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
        print("🚀 LEGAL SUITE V72 (Jurisprudence Architect - 5 Major Upgrades)...")
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

        # Gelişmiş Konsolidasyon (V72)
        if self.memory_manager:
            self.memory_manager.consolidate_principles_v72()

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