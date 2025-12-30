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

class LegalPDF(FPDF):

    def header(self):
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 10, "HUKUKI DEGERLENDIRME RAPORU", ln=True)
        self.ln(2)

    def section(self, title: str):
        self.ln(4)
        self.set_font("DejaVu", "B", 11)
        self.multi_cell(0, 6, title)
        self.ln(1)

    def paragraph(self, text: str):
        self.set_font("DejaVu", "", 10)
        for line in text.split("\n"):
            self.multi_cell(0, 5, line)
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

        # Unicode font
        pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf", uni=True)

        # 1️⃣ Hızlı Özet
        pdf.section("Hizli Ozet")
        pdf.paragraph(
            f"Hakim ilk refleksi: {judge_reflex.tendency}\n"
            f"Genel guc skoru: {judge_reflex.score}/100\n"
            f"Tespit edilen tereddut sayisi: {len(judge_reflex.doubts)}"
        )

        # 2️⃣ Hakim Tereddütleri
        pdf.section("Hakimin Tereddutleri")
        if judge_reflex.doubts:
            for d in judge_reflex.doubts:
                pdf.paragraph(f"- {d}")
        else:
            pdf.paragraph("Belirgin bir tereddut tespit edilmemistir.")

        # 3️⃣ Persona Görüşleri
        pdf.section("Taraf Gorüsleri")
        for p in persona_outputs:
            pdf.paragraph(f"{p.role}:\n{p.response}")

        # 4️⃣ Güçlendirme & Aksiyon Planı
        pdf.section("Guclendirme ve Aksiyon Plani")
        for a in actions:
            pdf.paragraph(
                f"{a.title}\n"
                f"{a.description}\n"
                f"Etki Puani: {a.impact_score}/10"
            )

        # 5️⃣ İncelenen Belgeler
        pdf.section("Incelenen Belgeler")
        for doc in documents:
            pdf.paragraph(
                f"{doc.get('source')} | "
                f"Güven: %{int(doc.get('confidence', 0) * 100)}"
            )

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
