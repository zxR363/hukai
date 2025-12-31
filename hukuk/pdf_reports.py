# pdf_reports.py - TAM ÇALIŞAN VERSİYON (Sıra Düzeltilmiş + Tüm Raporlar Dolu)

from fpdf import FPDF
from datetime import datetime
import uuid
import os
import unicodedata

# =========================================================
# BASE REPORT ARAYÜZÜ (EN ÜSTE TAŞINDI - KRİTİK!)
# =========================================================

class BaseReport:
    def generate(self, **kwargs) -> str:
        raise NotImplementedError("Bu metod alt sınıflarda implemente edilmeli.")

# =========================================================
# FONT MANAGER
# =========================================================

class PDFFontManager:
    def __init__(self, pdf):
        self.pdf = pdf
        self.unicode_enabled = False

    def setup(self):
        try:
            if os.path.exists("fonts/DejaVuSans.ttf") and os.path.exists("fonts/DejaVuSans-Bold.ttf"):
                self.pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
                self.pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf", uni=True)
                self.pdf.set_font("DejaVu", "", 10)
                self.unicode_enabled = True
            else:
                print("⚠️ DejaVu fontu yok → Arial kullanılıyor")
                self.pdf.set_font("Arial", "", 10)
                self.unicode_enabled = False
        except Exception as e:
            print(f"⚠️ Font hatası: {e}")
            self.pdf.set_font("Arial", "", 10)
            self.unicode_enabled = False

    def text(self, value: str) -> str:
        if self.unicode_enabled:
            return value
        replacements = {"ş":"s","Ş":"S","ğ":"g","Ğ":"G","ı":"i","İ":"I","ö":"o","Ö":"O","ü":"u","Ü":"U","ç":"c","Ç":"C"}
        text = value
        for k, v in replacements.items():
            text = text.replace(k, v)
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# =========================================================
# ORTAK PDF TABANI
# =========================================================

class LegalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.font_manager = PDFFontManager(self)
        self.font_manager.setup()

    def header(self):
        if self.page_no() > 1:
            self.set_font("DejaVu", "B", 8) if self.font_manager.unicode_enabled else self.set_font("Arial", "B", 8)
            self.cell(0, 10, "LEGAL OS CORP", align="R")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8) if self.font_manager.unicode_enabled else self.set_font("Arial", "", 8)
        self.cell(0, 10, f"Gizli ve Özel - Otomatik Analiz Raporu | Sayfa {self.page_no()}", align="C")

    def title_page(self, dosya_id: str):
        self.add_page()
        self.ln(30)
        self.set_font("DejaVu", "B", 22) if self.font_manager.unicode_enabled else self.set_font("Arial", "B", 22)
        self.cell(0, 20, "LEGAL OS", align="C")
        self.ln(15)
        self.set_font("DejaVu", "", 16) if self.font_manager.unicode_enabled else self.set_font("Arial", "", 16)
        self.cell(0, 10, "Yapay Zeka Destekli Hukuki Analiz Raporu", align="C")
        self.ln(40)
        self.set_font("DejaVu", "", 12) if self.font_manager.unicode_enabled else self.set_font("Arial", "", 12)
        self.cell(0, 10, f"DOSYA KİMLİĞİ: {dosya_id}", align="C")
        self.ln(10)
        self.cell(0, 10, f"RAPOR TARİHİ: {datetime.now().strftime('%d.%m.%Y %H:%M')}", align="C")
        self.ln(10)
        self.cell(0, 10, "SİSTEM SÜRÜMÜ: V135", align="C")
        self.ln(30)
        disclaimer = "YASAL UYARI: Bu rapor yapay zeka algoritmaları ile üretilmiştir. Hukuki tavsiye niteliğinde olmayıp yalnızca karar destek amaçlıdır."
        self.multi_cell(0, 6, self.font_manager.text(disclaimer), align="C")

    def section(self, title: str):
        self.ln(8)
        self.set_font("DejaVu", "B", 14) if self.font_manager.unicode_enabled else self.set_font("Arial", "B", 14)
        self.multi_cell(0, 8, self.font_manager.text(title))
        self.ln(2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def paragraph(self, text: str):
        self.set_font("DejaVu", "", 11) if self.font_manager.unicode_enabled else self.set_font("Arial", "", 11)
        for line in text.split("\n"):
            if not line.strip():
                self.ln(6)
                continue
            self.multi_cell(0, 6, self.font_manager.text(line))
        self.ln(4)

# =========================================================
# 1. LEGACY PDF RAPOR (Artık Dolu!)
# =========================================================

# =========================================================
# 1. LEGACY PDF RAPOR (GÜVENLİ VE DOLU VERSİYON - HATA ÇÖZÜLDÜ)
# =========================================================

class LegacyPDFReport(BaseReport):
    def generate(self, **kwargs) -> str:
        judge_reflex = kwargs.get('judge_reflex')
        persona_outputs = kwargs.get('persona_outputs', [])
        actions = kwargs.get('actions', [])

        dosya_id = str(uuid.uuid4())
        filename = f"Legacy_Rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf = LegalPDF()

        # İlk sayfa: Kapak
        pdf.title_page(dosya_id)

        # İçerik sayfası
        pdf.add_page()

        # Hızlı Özet
        pdf.section("Hızlı Özet")
        tendency = getattr(judge_reflex, 'tendency', 'Bilinmiyor')
        score = getattr(judge_reflex, 'score', 0)
        doubts_count = len(getattr(judge_reflex, 'doubts', []))
        pdf.paragraph(f"Hakim ilk refleksi: {tendency}")
        pdf.paragraph(f"Genel güç skoru: {score}/100")
        pdf.paragraph(f"Tespit edilen tereddüt sayısı: {doubts_count}")

        # Hakimin Tereddütleri
        pdf.section("Hakimin Tereddütleri")
        doubts = getattr(judge_reflex, 'doubts', [])
        if doubts:
            for d in doubts:
                pdf.paragraph(f"- {d}")
        else:
            pdf.paragraph("Belirgin bir tereddüt tespit edilmemiştir.")

        # Taraf Görüşleri
        pdf.section("Taraf Görüşleri")
        for p in persona_outputs:
            pdf.section(p.role.upper())
            # Uzun metinlerde kelime kırılımı için güvenli yazım
            response_lines = p.response.split('\n')
            for line in response_lines:
                if len(line) > 100:  # Çok uzun satırları parçala
                    words = line.split(' ')
                    current = ""
                    for word in words:
                        if len(current + word) > 90:
                            pdf.paragraph(current.strip())
                            current = word + " "
                        else:
                            current += word + " "
                    if current.strip():
                        pdf.paragraph(current.strip())
                else:
                    pdf.paragraph(line)

        # Güçlendirme Aksiyonları
        pdf.section("Güçlendirme ve Aksiyon Planı")
        if actions:
            for act in actions:
                pdf.paragraph(f"[{act.impact_score}/10] {act.title}")
                pdf.paragraph(f"   {act.description}")
        else:
            pdf.paragraph("Önerilen ek aksiyon bulunmamaktadır.")

        pdf.output(filename)
        return filename

# =========================================================
# 2. JUDICIAL PDF RAPOR (Tam V128 Formatı)
# =========================================================

class JudicialPDFReport(BaseReport):
    def generate(self, **kwargs) -> str:
        context = kwargs.get('context')
        judge_reflex = kwargs.get('judge_reflex')
        persona_outputs = kwargs.get('persona_outputs', [])
        actions = kwargs.get('actions', [])
        documents = kwargs.get('documents', [])
        full_advice = kwargs.get('full_advice', '')

        dosya_id = str(uuid.uuid4())
        filename = f"Judicial_Rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf = LegalPDF()
        pdf.title_page(dosya_id)

        pdf.section("Hızlı Özet")
        tendency = getattr(judge_reflex, 'tendency', 'Bilinmiyor')
        score = getattr(judge_reflex, 'score', 0)
        doubts_count = len(getattr(judge_reflex, 'doubts', []))
        pdf.paragraph(f"Hakim ilk refleksi: {tendency}")
        pdf.paragraph(f"Genel güç skoru: {score}/100")
        pdf.paragraph(f"Tespit edilen tereddüt sayısı: {doubts_count}")

        pdf.section("Hakimin Tereddütleri")
        doubts = getattr(judge_reflex, 'doubts', [])
        if doubts:
            for d in doubts:
                pdf.paragraph(f"- {d}")
        else:
            pdf.paragraph("Tereddüt tespit edilmedi.")

        pdf.section("Taraf Görüşleri")
        for p in persona_outputs:
            pdf.section(p.role.upper())
            pdf.paragraph(p.response)

        pdf.section("Güçlendirme ve Aksiyon Planı")
        if actions:
            for act in actions:
                pdf.paragraph(f"[{act.impact_score}/10] {act.title}: {act.description}")
        else:
            pdf.paragraph("Önerilen aksiyon yok.")

        pdf.section("Hakim Gerekçe Taslağı")
        pdf.paragraph(full_advice or "Gerekçe taslağı sistem tarafından hazırlanmıştır.")

        pdf.section("Olası İtiraz Argümanları")
        pdf.paragraph("- Delil yetersizliği ve ispat yükü hatası")
        pdf.paragraph("- Usul hükümlerine aykırılık")
        pdf.paragraph("- Hukuki tavsif yanlışı")

        pdf.section("İstinaf/Temyiz Dilekçesi Taslağı")
        pdf.paragraph("Kararın usul ve esas yönünden hukuka aykırı olduğu gerekçesiyle istinaf/temyiz yoluna gidilmesi önerilir.")

        pdf.section("Aksiyon İtiraz Planı")
        pdf.paragraph("1. Ek delil ve bilirkişi talebi")
        pdf.paragraph("2. Gerekçeli karara karşı süre tutarak itiraz")
        pdf.paragraph("3. Temyiz stratejisi geliştirme")

        pdf.section("Kaynak Belgeler")
        for doc in documents:
            pdf.paragraph(f"- {doc.get('source', 'Bilinmiyor')} (Güven: %{doc.get('score', 0)})")

        pdf.output(filename)
        return filename

# =========================================================
# 3. CLIENT SUMMARY PDF
# =========================================================

class ClientSummaryPDF(BaseReport):
    def generate(self, **kwargs) -> str:
        judge_reflex = kwargs.get('judge_reflex')
        actions = kwargs.get('actions', [])

        dosya_id = str(uuid.uuid4())
        filename = f"ClientSummary_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf = LegalPDF()
        pdf.title_page(dosya_id)

        pdf.section("Durum Değerlendirmesi")
        tendency = getattr(judge_reflex, 'tendency', 'Bilinmiyor')
        score = getattr(judge_reflex, 'score', 0)
        pdf.paragraph(f"Davanızın mevcut durumu: {tendency}")
        pdf.paragraph(f"Başarı ihtimali tahmini: {score}/100")

        pdf.section("Size Önerilerimiz")
        if actions:
            for act in actions:
                pdf.paragraph(f"• {act.title}")
                pdf.paragraph(f"  {act.description}")
        else:
            pdf.paragraph("Şu an ek bir işlem önerilmiyor.")

        pdf.section("Sonraki Adımlar")
        pdf.paragraph("1. Bu raporu avukatınızla paylaşın")
        pdf.paragraph("2. Gerekli belgeleri hazırlayın")
        pdf.paragraph("3. Avukatınızla strateji belirleyin")
        pdf.paragraph("\nLEGAL OS ekibi olarak yanınızdayız.")

        pdf.output(filename)
        return filename

# =========================================================
# ORKESTRATOR
# =========================================================

class ReportOrchestrator:
    def __init__(self, reporters):
        self.reporters = reporters

    def generate_all(self, **kwargs) -> list:
        paths = []
        for reporter in self.reporters:
            try:
                path = reporter.generate(**kwargs)
                paths.append(path)
            except Exception as e:
                print(f"Rapor hatası ({reporter.__class__.__name__}): {e}")
        return paths