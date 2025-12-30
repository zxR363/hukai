import os
import sys
from dataclasses import dataclass
from typing import List

# Dosya yollarını kontrol et
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from pdf_reports import JudicialPDFReport, ReportOrchestrator

    print("✅ pdf_reports modülü başarıyla içe aktarıldı.")
except ImportError as e:
    print(f"❌ HATA: pdf_reports modülü bulunamadı. ({e})")
    sys.exit(1)


# ---------------------------------------------------------
# MOCK VERİ YAPILARI (V132'deki Dataclass'ların Taklidi)
# ---------------------------------------------------------
@dataclass
class MockJudgeReflex:
    tendency: str
    score: int
    doubts: List[str]


@dataclass
class MockPersonaResponse:
    role: str
    response: str


@dataclass
class MockStrengtheningAction:
    title: str
    description: str
    impact_score: int
    related_doubt: str = "Test"


# ---------------------------------------------------------
# TEST VERİLERİ (Türkçe Karakter Zorlamalı)
# ---------------------------------------------------------
print("\n🛠️  Test verileri hazırlanıyor...")

# 1. Hakim Refleksi
mock_reflex = MockJudgeReflex(
    tendency="KABUL EĞİLİMLİ (Şartlı)",
    score=85,
    doubts=[
        "Davacı tarafın 'işçi alacağı' iddiası ispatlanmalı.",
        "Özellikle 'Çalışma Bakanlığı' kayıtları eksik."
    ]
)

# 2. Persona Çıktıları
mock_personas = [
    MockPersonaResponse(
        role="DAVACI VEKİLİ",
        response="Müvekkilimiz 'ağır şartlarda' çalışmıştır. İspat yükü işverendedir."
    ),
    MockPersonaResponse(
        role="DAVALI VEKİLİ",
        response="İddialar asılsızdır. Zaman aşımı defi (itirazı) mevcuttur."
    )
]

# 3. Aksiyonlar
mock_actions = [
    MockStrengtheningAction(
        title="Tanık Dinletilmesi",
        description="İş yeri çalışma şartlarını bilen 2 şahit mahkemeye sunulmalı.",
        impact_score=8
    )
]

# 4. Belgeler
mock_docs = [
    {"source": "Yargıtay 9. HD 2023/12345", "confidence": 0.95, "type": "EMSAL"},
    {"source": "TMK Madde 6", "confidence": 1.0, "type": "MEVZUAT"}
]


# ---------------------------------------------------------
# TEST ÇALIŞTIRMA
# ---------------------------------------------------------
def run_test():
    print("\n🚀 Raporlama Orkestratörü Test Ediliyor...")

    # Font kontrolü (Bilgilendirme amaçlı)
    font_path = os.path.join(current_dir, "fonts", "DejaVuSans.ttf")
    if os.path.exists(font_path):
        print(f"ℹ️  Bilgi: DejaVu fontu bulundu ({font_path}). Unicode çıktı bekleniyor.")
    else:
        print("⚠️  Uyarı: Font dosyası bulunamadı. Sistem 'Arial' fallback modunda çalışacak (ASCII normalize).")

    try:
        # Orkestratörü başlat
        orchestrator = ReportOrchestrator(
            reporters=[JudicialPDFReport()]
        )

        # Raporu üret
        generated_files = orchestrator.generate_all(
            context=None,  # PDF raporunda doğrudan kullanılmıyor, None geçilebilir
            judge_reflex=mock_reflex,
            persona_outputs=mock_personas,
            actions=mock_actions,
            documents=mock_docs
        )

        print("\n✅ İŞLEM BAŞARILI!")
        print("--------------------------------------------------")
        for f in generated_files:
            if os.path.exists(f):
                print(f"📄 Oluşturulan Dosya: {f} (Boyut: {os.path.getsize(f)} bytes)")
            else:
                print(f"❌ Dosya oluşturulamadı: {f}")
        print("--------------------------------------------------")
        print("Lütfen oluşturulan PDF dosyasını açıp Türkçe karakterleri (İ, ş, ğ) kontrol edin.")

    except Exception as e:
        print(f"\n❌ TEST SIRASINDA HATA OLUŞTU:")
        print(e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_test()