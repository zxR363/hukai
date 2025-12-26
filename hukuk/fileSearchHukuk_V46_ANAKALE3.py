import sys
import os
import re
import uuid
import time
from multiprocessing import Pool, cpu_count

# --------------------------------------------------
# 📦 IMPORTLAR
# --------------------------------------------------
import fitz  # PyMuPDF (V44 İÇİN GEREKLİ - PDF SONU OKUMA)
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_community.document_loaders import PyMuPDFLoader  # Loader düzeltmesi

# UTF-8 Ayarı
sys.stdout.reconfigure(encoding="utf-8")

# ================== AYARLAR (V45 İLE AYNI) ==================
SOURCES = {
    "mevzuat": {
        "folder": "mevzuatlar",
        "collection": "legal_statutes_v43",
        "desc": "MEVZUAT"
    },
    "emsal": {
        "folder": "belgeler",
        "collection": "legal_precedents_v43",
        "desc": "EMSAL KARAR"
    }
}

QDRANT_PATH = "qdrant_db_master"
EMBEDDING_MODEL = "nomic-embed-text"

SEARCH_LIMIT_PER_SOURCE = 30
SCORE_THRESHOLD = 0.40
LLM_RERANK_LIMIT = 10


# ==================================================
# 1️⃣ MODÜL: PDF SONU OKUYUCU (V44 İLE AYNI)
# ==================================================
def extract_pdf_conclusion(file_path, char_limit=2500):
    """
    Vektör araması dosyanın başını bulsa bile, bu fonksiyon
    fiziksel dosyaya gidip PDF'in SON SAYFALARINI okur.
    Çünkü Yargıtay kararları genelde en sonda 'HÜKÜM:...' der.
    """
    try:
        if not os.path.exists(file_path):
            return "[Dosya bulunamadı, fiziksel okuma yapılamadı.]"

        doc = fitz.open(file_path)
        total_pages = len(doc)
        text = ""

        # Son 2 sayfayı hedefle (Karar ve Sonuç genelde buradadır)
        start_page = max(0, total_pages - 2)

        for i in range(start_page, total_pages):
            text += doc[i].get_text()

        doc.close()

        # Metnin son X karakterini temizleyip döndür
        return text[-char_limit:]
    except Exception as e:
        return f"[Karar kısmı okunamadı: {e}]"


# ==================================================
# 2️⃣ MEVCUT ARAÇLAR (V45 İLE AYNI)
# ==================================================
def worker_embed_batch(args):
    texts, model_name = args
    embedder = OllamaEmbeddings(model=model_name)
    return embedder.embed_documents(texts)


def clean_text(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_expanded_queries(llm, story, topic):
    print("   ↳ 🧠 Sorgu Genişletiliyor...")
    prompt = f"""GÖREV: Hukuki olayı analiz et.
OLAY: "{story}"
ODAK: {topic}
Arama motoru için 3 farklı bakış açısıyla (Hakim, Avukat, Mevzuat) 3 kısa cümle yaz. Başlık koyma."""
    try:
        res = llm.invoke(prompt).content
        return [line.strip() for line in res.splitlines() if len(line) > 10][:3]
    except:
        return [story]


def check_relevance_judge_smart(llm, user_query, user_filter, negative_keywords, document_text, source_name):
    found_negative = None
    if negative_keywords:
        doc_lower = document_text.lower()
        for bad in negative_keywords:
            pattern = re.compile(rf"\b{re.escape(bad)}\b")
            if pattern.search(doc_lower):
                found_negative = bad
                break

    if found_negative:
        prompt = f"""
SEN HUKUK EDİTÖRÜSÜN.
Sorgu: "{user_query}" ({user_filter}).
Belgede yasaklı "{found_negative}" kelimesi geçiyor.
Belge: "{document_text[:600]}..."
Bu kelime konuyu tamamen saptırıyor mu (RED)? Yoksa bağlam uygun mu (KABUL)?
CEVAP: [RED] veya [KABUL] ve sebebi.
"""
        res = llm.invoke(prompt).content.strip()
        if "RED" in res:
            return False, f"⛔ YASAKLI KELİME ({found_negative}): {res}"

    scope = f"Odak: {user_filter}" if user_filter else "Genel Hukuk"
    prompt_gen = f"""
SEN HUKUKÇUSUN.
Sorgu: "{user_query}"
Bağlam: {scope}
Belge: "{document_text[:600]}..."
Bu belge bu konuya delil olabilir mi?
CEVAP: [EVET] veya [HAYIR] ve sebebi.
"""
    res = llm.invoke(prompt_gen).content.strip()
    return "EVET" in res.upper(), res


# ==================================================
# 3️⃣ INDEXING ENGINE (V45 İLE AYNI)
# ==================================================
def run_indexing_v44():
    client = QdrantClient(path=QDRANT_PATH)

    for key, config in SOURCES.items():
        collection_name = config["collection"]
        folder_path = config["folder"]

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"⚠️ '{folder_path}' klasörü oluşturuldu.")
            continue

        if not client.collection_exists(collection_name):
            print(f"⚙️ '{collection_name}' ({config['desc']}) kutusu oluşturuluyor...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        indexed_files = set()
        offset = None
        while True:
            points, offset = client.scroll(collection_name, offset=offset, limit=100, with_payload=True,
                                           with_vectors=False)
            for p in points:
                if 'source' in p.payload: indexed_files.add(p.payload['source'])
            if offset is None: break

        files_on_disk = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        new_files = [f for f in files_on_disk if f not in indexed_files]

        if not new_files:
            print(f"✅ {config['desc']} güncel.")
            continue

        print(f"♻️ {config['desc']} için {len(new_files)} yeni dosya işleniyor...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_texts = []
        all_metadatas = []

        for filename in new_files:
            try:
                loader = PyMuPDFLoader(os.path.join(folder_path, filename))
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                for c in chunks:
                    clean_content = clean_text(c.page_content)
                    all_texts.append(clean_content)
                    all_metadatas.append({
                        "source": filename,
                        "type": config['desc'],
                        "page": c.metadata.get("page", 0) + 1
                    })
                print(f"   📄 Okundu: {filename}")
            except Exception as e:
                print(f"   ⚠️ Hata: {filename} - {e}")

        if not all_texts: continue

        print(f"   🚀 Vektörleştiriliyor ({len(all_texts)} parça)...")
        num_cores = cpu_count()
        batch_size = (len(all_texts) // num_cores) + 1
        batches = []
        for i in range(0, len(all_texts), batch_size):
            batches.append((all_texts[i:i + batch_size], EMBEDDING_MODEL))

        all_vectors = []
        with Pool(processes=num_cores) as pool:
            results = pool.map(worker_embed_batch, batches)
            for res in results: all_vectors.extend(res)

        print(f"   💾 {collection_name} kutusuna yazılıyor...")
        points = []
        for i, (vec, meta, txt) in enumerate(zip(all_vectors, all_metadatas, all_texts)):
            payload = {"page_content": txt, "source": meta["source"], "page": meta["page"], "type": meta["type"]}
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, txt + meta["source"] + collection_name))
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        batch_size_upload = 64
        for i in range(0, len(points), batch_size_upload):
            client.upsert(collection_name, points[i:i + batch_size_upload])

    print("✅ Tüm indeksleme tamamlandı.")
    return True


# ==================================================
# 4️⃣ PDF REPORT
# ==================================================
class LegalReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'HUKUKI ANALIZ RAPORU', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C');
        self.ln(5)

    def footer(self):
        self.set_y(-15);
        self.set_font('helvetica', 'I', 8);
        self.cell(0, 10, f'Sayfa {self.page_no()}', align='C')


def create_pdf_report(user_story, valid_docs, advice_text, filename="Hukuki_Rapor_V46.pdf"):
    pdf = LegalReport();
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
        pdf.set_font(style='I', size=8);
        pdf.multi_cell(w=pdf.epw, h=4, text=clean(f"SEBEP: {doc['reason']}"));
        pdf.ln(2)

    pdf.add_page();
    pdf.set_font(style='B', size=12);
    pdf.cell(0, 10, clean("3. HUKUKI GORUS:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(style='', size=10);
    pdf.multi_cell(w=pdf.epw, h=6, text=clean(advice_text))
    try:
        pdf.output(filename); print(f"\n📄 Rapor Hazır: {filename}")
    except:
        pass


# ==================================================
# 5️⃣ ANA MOTOR (V46: FULL DISCLOSURE MODE)
# ==================================================
def main():
    print("🚀 LEGAL SUITE V46 (Full Disclosure: All Evidence Mode)...")

    if not run_indexing_v44():
        sys.exit()

    llm = ChatOllama(model="qwen2.5", temperature=0.1)
    dense_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    client = QdrantClient(path=QDRANT_PATH)

    print("\n✅ SİSTEM HAZIR. (Çıkış: 'q')")

    while True:
        print("-" * 60)
        story = input("📝 Olay: ")
        if story.lower() == "q": break
        topic = input("🎯 Odak: ")
        neg_input = input("🚫 Yasaklı: ")
        negatives = [w.strip().lower() for w in neg_input.split(",")] if neg_input else []

        expanded = generate_expanded_queries(llm, story, topic)
        full_query = f"{story} {topic} " + " ".join(expanded)
        print(f"   ✓ Sorgu: {len(full_query)} karakter")

        print("\n🔍 Belgeler Taranıyor (Dual Search)...")
        query_vector = dense_embedder.embed_query(full_query)
        all_candidates = []

        for key, config in SOURCES.items():
            results = client.query_points(
                collection_name=config["collection"],
                query=query_vector,
                limit=40
            ).points
            for hit in results:
                if 'type' not in hit.payload: hit.payload['type'] = config['desc']
                all_candidates.append(hit)

        unique_docs = {}
        for hit in all_candidates:
            if hit.score < SCORE_THRESHOLD: continue
            key = f"{hit.payload['source']}_{hit.payload['page']}"
            if key not in unique_docs or hit.score > unique_docs[key].score:
                unique_docs[key] = hit

        candidates = sorted(unique_docs.values(), key=lambda x: x.score, reverse=True)[:LLM_RERANK_LIMIT]
        if not candidates: print("🔴 Skor eşiğini geçen belge bulunamadı."); continue

        print("\n⚖️  Akıllı Yargıç Değerlendiriyor:")
        valid_docs = []

        for hit in candidates:
            doc_text = hit.payload['page_content']
            source = hit.payload['source']
            page = hit.payload['page']
            type_desc = hit.payload['type']

            is_ok, reason = check_relevance_judge_smart(llm, story, topic, negatives, doc_text, source)
            norm_score = min(max(hit.score, 0), 1) * 100

            icon = "✅" if is_ok else "❌"
            print(f"{icon} [{type_desc}] {source:<20} | Güven: %{norm_score:.1f}")

            if is_ok:
                extra_context = ""
                # V44 Özelliği: Smart Stitching
                if type_desc == "EMSAL KARAR":
                    real_path = os.path.join(SOURCES["emsal"]["folder"], source)
                    verdict = extract_pdf_conclusion(real_path)
                    # Burayı daha belirgin yapıyoruz
                    extra_context = f"\n\n🛑 BU BELGE BİR MAHKEME KARARIDIR. İŞTE SONUCU ({source}):\n{verdict}\n🛑 KARAR SONU."

                valid_docs.append({
                    "source": source,
                    "page": page,
                    "type": type_desc,
                    "text": doc_text + extra_context,
                    "score": hit.score,
                    "reason": reason
                })

        if not valid_docs: print("🔴 Yargıç tüm belgeleri eledi."); continue

        context_str = ""
        for d in valid_docs:
            context_str += f"""
>>> TÜR: [{d['type']}]
DOSYA ADI: {d['source']}
SAYFA: {d['page']}
NEDEN SEÇİLDİ: {d['reason']}
İÇERİK:
{d['text']}
=========================================
"""

        print("\n🧑‍⚖️  AVUKAT YAZIYOR (Full Disclosure Mode)...")

        # --- V46 PROMPT GÜNCELLEMESİ (HEPSİNİ LİSTELE EMRİ) ---
        prompt = f"""
SEN KIDEMLİ BİR HUKUKÇUSUN.
Aşağıdaki "DELİLLER" kısmındaki metinleri kullanarak olayı analiz et.

OLAY: "{story}"
ODAK: "{topic}"

DELİLLER:
{context_str}

⚠️ KIRMIZI ÇİZGİLER VE KURALLAR (BUNLARA UYMAZSAN İŞLEM GEÇERSİZDİR):

1. **KAYNAK AYRIMI:** - [MEVZUAT] türündeki belgeleri SADECE "Mevzuat Dayanakları" başlığında kullan.
   - [EMSAL KARAR] türündeki belgeleri SADECE "İlgili Emsal Kararlar" başlığında kullan.
   - ASLA bir Mevzuat belgesini (örneğin Miras Hukuku.pdf) Emsal Karar gibi sunma!

2. **FORMAT:**

   A. MEVZUAT DAYANAKLARI
      - Deliller listesindeki [MEVZUAT] etiketli belgelerin HEPSİNİ tek tek maddeler halinde yaz. Hiçbirini atlama.
      - Kanun maddelerini ve hukuki teoriyi buraya yaz.
      - Kaynak: (Dosya Adı, Sayfa)

   B. İLGİLİ EMSAL KARARLAR (Yargıtay/İstinaf)
      - Deliller listesindeki [EMSAL KARAR] etiketli belgelerin HEPSİNİ tek tek özetle. Hiçbirini atlama.
      - Deliller listesinde [EMSAL KARAR] yazan belgeler var mı? Varsa buraya yaz.
      - Yoksa "İncelenen belgeler arasında doğrudan bir emsal karar dosyası bulunamadı" de. ASLA kitaplardan örnek uydurma.
      - Eğer [EMSAL KARAR] varsa, metnin sonundaki "OTOMATİK EKLENEN KARAR SONUCU" veya "HÜKÜM" kısmını özetle.
      - Kaynak: (Dosya Adı, Sayfa)

   C. SONUÇ VE TAVSİYE
      - Net ve uygulanabilir bir yol haritası.

3. **ASLA "X Dosyası" DEME:** Dosyanın tam adını (örn: buyuk7.pdf) kullan.

ANALİZİ BAŞLAT:
"""
        full_res = ""
        for chunk in llm.stream(prompt):
            c = chunk.content;
            full_res += c;
            print(c, end="", flush=True)
        print("\n")

        print("\n" + "=" * 20 + " 📚 KULLANILAN KAYNAKLAR " + "=" * 20)
        for d in valid_docs:
            print(f"• [{d['type']}] {d['source']} (Sf. {d['page']})")
        print("=" * 64)

        create_pdf_report(story, valid_docs, full_res)


if __name__ == "__main__":
    main()