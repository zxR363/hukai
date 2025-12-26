import asyncio
import random

class LegalSearchEngine:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback

    async def alog(self, msg: str):
        if self.log_callback:
            await self.log_callback(msg)

    async def run_analysis(self, story: str, topic: str, negatives: list):
        # 1. Start
        await self.alog("-" * 60)
        await self.alog(f"📝 Olay: {story}")
        await self.alog(f"🎯 Odak: {topic}")
        await asyncio.sleep(0.5)

        # 2. Search Simulation
        await self.alog("\n🔍 Belgeler Taranıyor (Dual Search - Aşama 1)...")
        await asyncio.sleep(1.0)
        await self.alog("   ✓ Sorgu Genişletildi: 145 karakter")
        await self.alog("   ✓ Vektör Arama Tamamlandı: 25 aday bulundu")

        # 3. Judge Simulation
        await self.alog("\n⚖️  Akıllı Yargıç Değerlendiriyor (Aşama 2: Rol Atama)...")
        
        # Generate some dummy docs
        docs = []
        doc_templates = [
            ("Yargitay_3_HD_2023_145.pdf", "EMSAL KARAR", "[EMSAL İLKE]", 92.5),
            ("Yargitay_12_CD_2022_89.pdf", "EMSAL KARAR", "[DOĞRUDAN DELİL]", 88.0),
            ("TBK_Madde_444.pdf", "MEVZUAT", "[EMSAL İLKE]", 95.0),
            ("Bilirkişi_Raporu_Örnek.pdf", "EMSAL KARAR", "[EMSAL İLKE]", 75.4),
            ("Anayasa_Mahkemesi_Karari.pdf", "EMSAL KARAR", "[DOĞRUDAN DELİL]", 82.1),
        ]

        for i, (src, type_desc, role, score) in enumerate(doc_templates):
            await asyncio.sleep(0.3) # Simulate processing time per doc
            reason = f"Bu belge, {topic} konusundaki {random.choice(['emsal niteliği', 'hukuki dayanağı', 'benzerlik derecesi'])} nedeniyle seçilmiştir."
            
            await self.alog(f"✅ [{type_desc}] {src} | Güven: %{score:.1f} | Rol: {role}")
            
            docs.append({
                "source": src,
                "page": i + 1,
                "type": type_desc,
                "role": role,
                "text": f"<h1>{src} İçeriği</h1><p>Bu bir simülasyon içeriğidir. {story} konusu ile ilgili önemli hukuki değerlendirmeler içermektedir.</p><p>LOREM IPSUM DOLOR SIT AMET...</p>",
                "score": score,
                "reason": reason
            })

        # 4. Writing Simulation
        await self.alog("\n🧑‍⚖️  AVUKAT YAZIYOR (Role-Aware Mode)...")
        await asyncio.sleep(1.5)
        
        advice = """
# HUKUKİ ANALİZ RAPORU

## A. MEVZUAT DAYANAKLARI
Bu olayda Türk Borçlar Kanunu Madde 444 ve ilgili yönetmelikler esas alınmalıdır.

## B. İLGİLİ EMSAL KARARLAR
**1. Yargıtay 3. Hukuk Dairesi 2023/145:**
Benzer bir uyuşmazlıkta mahkeme, kiracının tahliyesine karar vermiştir.

**2. Yargıtay 12. Ceza Dairesi 2022/89:**
Burada suçun maddi unsurlarının oluşmadığına hükmedilmiştir.

## C. SONUÇ VE TAVSİYE
Müvekkilinizin durumu, yukarıdaki emsal kararlar ışığında değerlendirildiğinde, davanın lehine sonuçlanma ihtimali yüksektir. Ancak delillerin sağlamlaştırılması gerekmektedir.
"""
        await self.alog("\n✅ Analiz Tamamlandı.")
        
        return advice, docs

# Helper to emulate original file's other exports if needed
def create_pdf_report_file(story, docs, advice, path):
    # Just create a dummy file
    with open(path, "w", encoding="utf-8") as f:
        f.write("DUMMY PDF CONTENT")
    return True
