# pdf_reports.py

from fpdf import FPDF
from datetime import datetime
from typing import List, Any


# =========================================================
# 1️⃣ ORTAK RAPOR ARAYÜZÜ
# =========================================================

class BaseReport:
    """
    Tüm PDF raporlarının uyması gereken arayüz.
    """

    def generate(self, **kwargs) -> str:
        raise NotImplementedError


# =========================================================
# 2️⃣ ORTAK PDF TABANI (UNICODE + TRUNCATION SAFE)
# =========================================================

# pdf_reports.py İÇİNDEKİ GÜNCEL LegalPDF SINIFI

class LegalPDF(FPDF):

    def header(self):
        # Header boş kalsın veya gerekirse ekleme yapılabilir
        pass

    def section(self, title: str, font_manager):
        self.ln(4)
        # Başlık için Bold (B) kullanımı
        if font_manager.unicode_enabled:
            self.set_font("DejaVu", "B", 11)
        else:
            self.set_font("Arial", "B", 11)

        # DÜZELTME BURADA: w=0 yerine w=self.epw kullanıyoruz
        # self.epw = Effective Page Width (Sayfa genişliği - Kenar boşlukları)
        self.multi_cell(w=self.epw, h=6, text=font_manager.text(title))
        self.ln(1)

    def paragraph(self, text: str, font_manager):
        # Metin için Normal font
        if font_manager.unicode_enabled:
            self.set_font("DejaVu", "", 10)
        else:
            self.set_font("Arial", "", 10)

        for line in text.split("\n"):
            # Boş satırları atla veya sadece boşluk bas
            if not line.strip():
                self.ln(5)
                continue

            # DÜZELTME BURADA: w=0 yerine w=self.epw
            # Ayrıca imleci garanti olarak sol kenara çekiyoruz (self.set_x)
            self.set_x(self.l_margin)
            self.multi_cell(w=self.epw, h=5, text=font_manager.text(line))

        self.ln(1)

# =========================================================
# 3️⃣ LEGACY PDF (ESKI SISTEMI EZMEZ)
# =========================================================

class LegacyPDFReport(BaseReport):
    """
    Mevcut PDF yapını buraya SARARSIN.
    İçini DEĞİŞTİRME.
    """

    def generate(self, **kwargs) -> str:
        """
        Burada eski generate_legacy_pdf(...) çağrılır.
        Şimdilik placeholder.
        """
        filename = f"Legacy_Rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, "Legacy PDF raporu (mevcut sistem)")
        pdf.output(filename)
        return filename


# =========================================================
# 4️⃣ JUDICIAL PDF (KIDEMLI AVUKAT IC RAPORU)
# =========================================================

class JudicialPDFReport(BaseReport):

    def generate(
        self,
        context: Any,
        judge_reflex: Any,
        persona_outputs: List[Any],
        actions: List[Any],
        documents: List[dict],
        **kwargs
    ) -> str:

        pdf = LegalPDF()
        pdf.add_page()

        # ✅ YENİ: FONT MANAGER KURULUMU
        font_manager = PDFFontManager(pdf)
        font_manager.setup()

        # Başlık (Manuel ekleme)
        if font_manager.unicode_enabled:
            pdf.set_font("DejaVu", "B", 12)
        else:
            pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, font_manager.text("HUKUKİ DEĞERLENDİRME RAPORU"), ln=True)
        pdf.ln(2)

        # 1️⃣ Hızlı Özet
        pdf.section("Hızlı Özet", font_manager)
        pdf.paragraph(
            f"Hakim ilk refleksi: {judge_reflex.tendency}\n"
            f"Genel güç skoru: {judge_reflex.score}/100\n"
            f"Tespit edilen tereddüt sayısı: {len(judge_reflex.doubts)}",
            font_manager
        )

        # 2️⃣ Hakim Tereddütleri
        pdf.section("Hakimin Tereddütleri", font_manager)
        if judge_reflex.doubts:
            for d in judge_reflex.doubts:
                pdf.paragraph(f"- {d}", font_manager)
        else:
            pdf.paragraph("Belirgin bir tereddüt tespit edilmemiştir.", font_manager)

        # 3️⃣ Persona Görüşleri
        pdf.section("Taraf Görüşleri", font_manager)
        for p in persona_outputs:
            pdf.paragraph(f"{p.role}:\n{p.response}", font_manager)

        # 4️⃣ Güçlendirme & Aksiyon Planı
        pdf.section("Güçlendirme ve Aksiyon Planı", font_manager)
        if actions:
            for a in actions:
                pdf.paragraph(
                    f"{a.title}\n"
                    f"{a.description}\n"
                    f"Etki Puanı: {a.impact_score}/10",
                    font_manager
                )
        else:
            pdf.paragraph("Önerilen ek aksiyon bulunmamaktadır.", font_manager)

        # 5️⃣ İncelenen Belgeler
        pdf.section("İncelenen Belgeler", font_manager)
        if documents:
            for doc in documents:
                pdf.paragraph(
                    f"{doc.get('source')} | "
                    f"Güven: %{int(doc.get('confidence', 0) * 100)}",
                    font_manager
                )
        else:
            pdf.paragraph("Belge verisi bulunamadı.", font_manager)

        filename = f"Judicial_Rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(filename)
        return filename


# =========================================================
# 5️⃣ MUSTERI OZET PDF (CLIENT-FRIENDLY)
# =========================================================

class ClientSummaryPDF(BaseReport):

    def generate(
        self,
        client_summary: Any,
        **kwargs
    ) -> str:

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)

        pdf.cell(0, 10, "MUSTERI BILGILENDIRME OZETI", ln=True)
        pdf.ln(4)

        pdf.multi_cell(0, 6, client_summary.case_overview)

        pdf.ln(3)
        pdf.cell(0, 8, "Guclu Yonler", ln=True)
        for s in client_summary.strengths:
            pdf.multi_cell(0, 6, f"- {s}")

        pdf.ln(2)
        pdf.cell(0, 8, "Olası Riskler", ln=True)
        for r in client_summary.risks:
            pdf.multi_cell(0, 6, f"- {r}")

        pdf.ln(2)
        pdf.cell(0, 8, "Sonraki Adimlar", ln=True)
        for n in client_summary.next_steps:
            pdf.multi_cell(0, 6, f"- {n}")

        pdf.ln(4)
        pdf.cell(
            0, 8,
            f"Genel Degerlendirme Seviyesi: {client_summary.confidence_level}",
            ln=True
        )

        filename = f"Musteri_Ozet_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(filename)
        return filename


# =========================================================
# 6️⃣ RAPOR ORKESTRATORU (AYNI ANDA COKLU PDF)
# =========================================================

class ReportOrchestrator:
    """
    Tek analizden birden fazla PDF üretir.
    """

    def __init__(self, reporters: List[BaseReport]):
        self.reporters = reporters

    def generate_all(self, **kwargs) -> List[str]:
        paths = []
        for reporter in self.reporters:
            path = reporter.generate(**kwargs)
            paths.append(path)
        return paths



##############FONT

# pdf_font_manager.py

import os
import unicodedata


class PDFFontManager:
    """
    Font güvenliği + Türkçe fallback yöneticisi
    """

    def __init__(self, pdf):
        self.pdf = pdf
        self.unicode_enabled = False

    def setup(self):
        """
        DejaVu varsa Unicode aktif
        Yoksa ASCII fallback
        """
        try:
            if (
                os.path.exists("fonts/DejaVuSans.ttf")
                and os.path.exists("fonts/DejaVuSans-Bold.ttf")
            ):
                self.pdf.add_font(
                    "DejaVu", "",
                    "fonts/DejaVuSans.ttf",
                    uni=True
                )
                self.pdf.add_font(
                    "DejaVu", "B",
                    "fonts/DejaVuSans-Bold.ttf",
                    uni=True
                )
                self.pdf.set_font("DejaVu", "", 10)
                self.unicode_enabled = True
            else:
                raise FileNotFoundError

        except Exception:
            # 🔒 Güvenli fallback
            self.pdf.set_font("Arial", "", 10)
            self.unicode_enabled = False

    def text(self, value: str) -> str:
        """
        Unicode yoksa Türkçe karakterleri sadeleştir
        """
        if self.unicode_enabled:
            return value

        return self._normalize_turkish(value)

    @staticmethod
    def _normalize_turkish(text: str) -> str:
        replacements = {
            "ş": "s", "Ş": "S",
            "ğ": "g", "Ğ": "G",
            "ı": "i", "İ": "I",
            "ö": "o", "Ö": "O",
            "ü": "u", "Ü": "U",
            "ç": "c", "Ç": "C",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # ekstra güvenlik
        return unicodedata.normalize("NFKD", text).encode(
            "ascii", "ignore"
        ).decode("ascii")
