import os
from dotenv import load_dotenv
from passlib.hash import bcrypt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

load_dotenv()
REPORT_DIR = os.getenv("REPORT_DIR", "E:\\genomics_project\\outputs\\reports")

def hash_password(p: str) -> str:
    return bcrypt.hash(p)

def verify_password(p: str, h: str) -> bool:
    return bcrypt.verify(p, h)

def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)

def generate_simple_pdf(pdf_path: str, title: str, lines: list[str]):
    ensure_dirs()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    for line in lines:
        c.drawString(40, y, line[:120])
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 50
    c.save()
