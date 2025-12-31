# test_pdf_reports.py
# Yeni pdf_reports.py'yi test etmek için bağımsız script
# Çalıştır: python test_pdf_reports.py

from datetime import datetime
import uuid

# pdf_reports.py'yi import et (aynı klasörde olmalı)
from pdf_reports import (
    LegacyPDFReport,
    JudicialPDFReport,
    ClientSummaryPDF,
    ReportOrchestrator
)

# Mock (sahte) veri hazırlıyoruz – gerçek sistemdeki gibi
class MockJudgeReflex:
    def __init__(self):
        self.tendency = "TEREDDÜTLÜ – BİLİRKİŞİ MUĞLAK"
        self.score = 25
        self.doubts = [
            "Zayıf içtihat/ilke tespiti",
            "Delil zincirinde belirsizlik"
        ]

class MockPersonaResponse:
    def __init__(self, role, response):
        self.role = role
        self.response = response

class MockAction:
    def __init__(self, impact_score, title, description):
        self.impact_score = impact_score
        self.title = title
        self.description = description

# Sahte veri
mock_context = None  # Kullanılmıyor ama kwargs için lazım

mock_judge_reflex = MockJudgeReflex()

mock_persona_outputs = [
    MockPersonaResponse("DAVACI VEKİLİ",
        "Davacı vekili olarak, mevcut delillerin davayı desteklediğini düşünüyorum. "
        "Ancak karşı tarafın olası itirazlarına karşı daha güçlü emsal kararlar sunulabilir."),
    MockPersonaResponse("DAVALI VEKİLİ",
        "Davalı vekili olarak, delil yetersizliği ve usul hataları nedeniyle davanın reddedilmesi gerektiğini savunuyorum. "
        "Yargıtay içtihatları bu yöndedir."),
    MockPersonaResponse("BİLİRKİŞİ",
        "Bilirkişi olarak, delil zincirinde bazı belirsizlikler tespit ettim. "
        "Ek inceleme ile daha net bir kanaate varılabilir.")
]

mock_actions = [
    MockAction(10, "Ek Emsal Karar Araştırması",
               "Zayıf içtihat tespitini güçlendirmek için Yargıtay 2. ve 3. Hukuk Dairesi kararları taranmalı."),
    MockAction(8, "Bilirkişi Raporu Talebi",
               "Mahkemeden ek bilirkişi incelemesi talep edilerek tereddüt giderilebilir.")
]

mock_documents = [
    {"source": "TürkMedeniKanunu.pdf", "score": 94.7},
    {"source": "buyuk2.pdf", "score": 96.7},
    {"source": "buyuk7.pdf", "score": 93.6},
]

mock_full_advice = (
    "Türk Medeni Kanunu md. 598 ve 605 hükümleri ile Yargıtay içtihatları ışığında, "
    "mirasçılık belgesi talebinde mirastan çıkarma (ıskat) bulunsa dahi Sulh Hukuk Mahkemesi görevlidir. "
    "Mirastan çıkarılan kişinin mirasçılık sıfatı tamamen kalkmaz; bu husus veraset ilamında açıklayıcı şekilde belirtilmelidir. "
    "Dosyada bu yönde yeterli delil bulunmamaktadır."
)

# Test verisi
test_kwargs = {
    "context": mock_context,
    "judge_reflex": mock_judge_reflex,
    "persona_outputs": mock_persona_outputs,
    "actions": mock_actions,
    "documents": mock_documents,
    "full_advice": mock_full_advice
}

print("PDF raporları üretiliyor...\n")

# Orchestrator ile üç rapor birden üret
orchestrator = ReportOrchestrator([
    LegacyPDFReport(),
    JudicialPDFReport(),
    ClientSummaryPDF()
])

produced_files = orchestrator.generate_all(**test_kwargs)

print("Üretilen PDF'ler:")
for file in produced_files:
    print(f"   ✅ {file}")

print("\nTest tamamlandı! Dosyaları klasörde kontrol edebilirsin.")
print("Legacy: Basit iç rapor")
print("Judicial: Tam detaylı rapor (gerekçe, itiraz, istinaf taslağı vs.)")
print("ClientSummary: Müşteriye verilecek sade özet")