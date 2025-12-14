from datetime import datetime
from typing import Any

from fpdf import FPDF

from obesity_cdss.config import settings
from obesity_cdss.web.utils import translate_single_value


class PDFReport(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 15)
        self.cell(
            0,
            10,
            "Relatório de Auxílio Diagnóstico para Obesidade",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )
        self.ln(5)

        self.set_line_width(0.5)
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0,
            10,
            "Este documento foi gerado por um modelo de Machine Learning "
            f"(Obesity CDSS v{settings.version}).",
            align="C",
        )
        self.ln(4)
        self.cell(
            0,
            10,
            "A decisão clínica final é de responsabilidade exclusiva do médico.",
            align="C",
        )


def generate_pdf_bytes(
    form_data: dict[str, Any], result: dict[str, Any], data_dict: dict[str, Any]
) -> bytes:
    """
    Generates a PDF report in memory and returns the bytes for download.
    Translates the raw data using the data dictionary.
    """
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # ==========================================================
    # DATE AND TIME
    # ==========================================================
    now = datetime.now().strftime("%d/%m/%Y às %H:%M")
    pdf.cell(0, 10, f"Data da Consulta: {now}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ==========================================================
    # PACIENT DATA
    # ==========================================================
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "DADOS DO PACIENTE", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)

    for key, value in form_data.items():
        if key not in data_dict:
            continue

        meta = data_dict[key]
        label = meta["description"]

        display_value = translate_single_value(data_dict, key, value)

        line_text = f"{label}: {display_value}"
        try:
            pdf.cell(
                0,
                7,
                line_text.encode("latin-1", "replace").decode("latin-1"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
        except Exception:
            pdf.cell(0, 7, f"{key}: {value}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # ==========================================================
    # MODEL RESULT
    # ==========================================================
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "RESULTADO DA ANÁLISE", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)

    target_meta = data_dict.get("obesity", {})
    raw_label = result["label"]
    translated_label = target_meta.get("values", {}).get(raw_label, raw_label)
    prob = result["probability"]

    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, pdf.get_y(), 190, 25, style="F")

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        8,
        f"Classificação: {translated_label}",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(
        0, 8, f"Probabilidade do Modelo: {prob:.1%}", new_x="LMARGIN", new_y="NEXT"
    )

    return bytes(pdf.output())
