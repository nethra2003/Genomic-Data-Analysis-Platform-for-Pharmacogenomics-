from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
from db_config import SessionLocal
from datetime import datetime
from fastapi import HTTPException
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from models import Report, Patient
from auth import router as auth_router
from fastapi import FastAPI, Request
import os, json


app = FastAPI(title="Genomic Data Analysis Backend")

# ===================================================
# CONFIG PATHS
# ===================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
RAW_VCF_DIR = BASE_DIR / "raw_vcf"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ===================================================
# CORS
# ===================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================
# 1️⃣  LIST FILES ENDPOINT
# ===================================================
@app.get("/list-files")
def list_files():
    """
    Return all available file names for each data category.
    """
    raw_files = [f.name for f in RAW_VCF_DIR.glob("*.vcf.gz")]
    variation_files = [f.name for f in DATA_DIR.glob("*_variants.csv")]
    annotated_files = (
        [f.name for f in DATA_DIR.glob("*_annotated_extracted.csv")] +
        [f.name for f in EXTRACTED_DIR.glob("*_annotated_*_extracted.csv")]
    )

    return {
        "raw": sorted(raw_files),
        "variation": sorted(variation_files),
        "annotated": sorted(annotated_files),
    }

ACTIVE_FILE_STORE = OUTPUTS_DIR / "active_files.json"

@app.post("/set-active-files")
async def set_active_files(request: Request):
    """
    Save the current uploaded VCF and generated CSV file names.
    """
    data = await request.json()
    ACTIVE_FILE_STORE.write_text(json.dumps(data, indent=2))
    return {"message": "Active file info saved ✅", "data": data}


@app.get("/get-active-files")
def get_active_files():
    """
    Retrieve the current active uploaded file and generated CSV.
    """
    if ACTIVE_FILE_STORE.exists():
        return json.loads(ACTIVE_FILE_STORE.read_text())
    return {"raw_file": None, "variation_file": None}

@app.get("/preview-variation")
def preview_variation(filename: str = "", nrows: int = 5):
    """
    Show variation preview (first nrows) only for the chromosome
    corresponding to the uploaded VCF file (like chr22).
    """
    import re

    if not filename:
        return {"error": "No filename provided."}

    # Detect chromosome from the uploaded VCF name
    match = re.search(r"chr(\d+|X|Y|MT)", filename, re.IGNORECASE)
    if not match:
        return {"error": f"Could not extract chromosome from filename: {filename}"}

    chrom = match.group(1)
    print(f"🧬 Detected chromosome: chr{chrom}")

    # Find variation CSVs that match this chromosome
    variation_files = list(DATA_DIR.glob(f"*chr{chrom}*_variants.csv"))

    if not variation_files:
        return {"error": f"No variation CSVs found for chromosome {chrom}."}

    dfs = []
    for f in variation_files:
        try:
            print(f"✅ Reading file: {f}")
            df = pd.read_csv(f, nrows=nrows)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f.name}: {e}")

    if not dfs:
        return {"error": f"Failed to read variation CSV for chr{chrom}."}

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Showing variation preview for chr{chrom}, rows: {len(combined_df)}")

    return {
        "chromosome": chrom,
        "rows_returned": len(combined_df),
        "data": combined_df.to_dict(orient="records"),
    }


import numpy as np

@app.get("/run-analysis")
def run_analysis(nrows: int = 5):
    """
    Load extracted annotated CSV(s) matching the currently active raw file.
    Filters to the correct chromosome (e.g., chr22) and returns a safe JSON preview.
    """
    import numpy as np

    # ✅ Step 1: Read active file
    active_file = None
    if ACTIVE_FILE_STORE.exists():
        try:
            active_data = json.loads(ACTIVE_FILE_STORE.read_text())
            active_file = active_data.get("raw_file")
        except Exception as e:
            print(f"⚠️ Could not read active file info: {e}")

    # ✅ Step 2: Extract chromosome (chr1, chr22, etc.)
    chr_label = None
    if active_file and "chr" in active_file.lower():
        parts = active_file.lower().split("chr")
        if len(parts) > 1:
            chr_number = ""
            for c in parts[1]:
                if c.isdigit() or c.lower() in ["x", "y", "m"]:
                    chr_number += c
                else:
                    break
            chr_label = f"chr{chr_number}"
            print(f"🧬 Detected chromosome: {chr_label}")

    # ✅ Step 3: Find extracted file(s)
    if chr_label:
        pattern = f"*{chr_label}*_annotated_*_extracted.csv"
        annotated_files = list(EXTRACTED_DIR.glob(pattern))
        print(f"🔍 Searching for extracted files with pattern: {pattern}")
    else:
        annotated_files = list(EXTRACTED_DIR.glob("*_annotated_*_extracted.csv"))
        print("⚠️ No active chromosome detected — loading all extracted files.")

    if not annotated_files:
        return {"error": f"No extracted annotated files found for {chr_label or 'any chromosome'}."}

    # ✅ Step 4: Read first few rows safely
    dfs = []
    for f in annotated_files:
        try:
            print(f"✅ Reading file: {f}")
            df = pd.read_csv(f, nrows=nrows)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f.name}: {e}")

    if not dfs:
        return {"error": "No readable annotated CSVs found."}

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Successfully loaded {len(dfs)} file(s) — previewing {len(combined_df)} rows.")

    # ✅ Step 5: Clean and serialize safely
    combined_df = combined_df.replace([np.inf, -np.inf], None).fillna("")
    try:
        json_data = json.loads(combined_df.to_json(orient="records", force_ascii=False))
        return {
            "chromosome": chr_label or "All",
            "total_files": len(dfs),
            "preview_rows": len(json_data),
            "records": json_data,
        }
    except Exception as e:
        print(f"⚠️ JSON encoding error: {e}")
        return {"error": "Failed to serialize data to JSON."}





@app.get("/list-training-genes")
def list_training_genes():
    TRAINING_PATH = OUTPUTS_DIR / "training_dataset.csv"
    if not TRAINING_PATH.exists():
        return {"error": f"Training dataset not found at {TRAINING_PATH}"}

    try:
        unique_genes = set()
        chunk_size = 100_000

        for chunk in pd.read_csv(TRAINING_PATH, usecols=["Gene"], chunksize=chunk_size, low_memory=False):
            chunk["Gene"] = chunk["Gene"].astype(str).str.strip()
            unique_genes.update(chunk["Gene"].dropna().unique())

            # stop early if already large list
            if len(unique_genes) > 5000:
                break

        genes_list = sorted(list(unique_genes))
        return {
            "total_genes": len(genes_list),
            "genes": genes_list,
            "source": str(TRAINING_PATH),
        }

    except Exception as e:
        import traceback
        print("❌ Error in /list-training-genes:", traceback.format_exc())
        return {"error": f"Internal server error: {str(e)}"}

@app.get("/list-drugs")
def list_drugs():
    import pandas as pd
    df = pd.read_csv("E:/genomics_project/outputs/training_dataset.csv", nrows=10000)
    drugs = sorted(df["Drug"].dropna().unique().tolist())
    return {"drugs": drugs}


@app.get("/run-report")
def run_report(gene: str = "", drug: str = "", nrows: int = 200):
    """
    Stream and filter pharmacogenomic report from training_dataset.csv.
    Supports large files efficiently and allows case-insensitive, partial search
    by gene or drug. Also saves the report in PostgreSQL linked to a patient.
    """

    TRAINING_PATH = OUTPUTS_DIR / "training_dataset.csv"

    # ✅ Step 1: Validate file existence
    if not TRAINING_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Training dataset not found at {TRAINING_PATH}")

    try:
        chunk_size = 100_000
        filtered_chunks = []
        total_rows = 0

        for chunk in pd.read_csv(TRAINING_PATH, chunksize=chunk_size, low_memory=False):
            # Standardize column names
            chunk.columns = [c.strip().replace(" ", "_") for c in chunk.columns]
            chunk.rename(columns={"gene": "Gene", "drug": "Drug"}, inplace=True, errors="ignore")

            if "Gene" not in chunk.columns and "Drug" not in chunk.columns:
                continue

            # ✅ Step 2: Case-insensitive, partial match filtering
            match = chunk.copy()
            if gene:
                match = match[match["Gene"].astype(str).str.contains(gene, case=False, na=False)]
            if drug:
                match = match[match["Drug"].astype(str).str.contains(drug, case=False, na=False)]

            if not match.empty:
                filtered_chunks.append(match)
                total_rows += len(match)

            if total_rows >= nrows:
                break  # Stop once enough rows collected

        # ✅ Step 3: Combine results
        if not filtered_chunks:
            search_label = gene or drug or "Unknown"
            raise HTTPException(status_code=404, detail=f"No matching records found for '{search_label}'")

        df = pd.concat(filtered_chunks, ignore_index=True).head(nrows)

        # ✅ Step 4: Select columns
        display_columns = [
            "CHROM", "POS", "Gene", "IMPACT", "Variant", "Drug",
            "Condition", "Dosage", "Response_Type", "Recommendation"
        ]
        available_cols = [c for c in display_columns if c in df.columns]
        display_df = df[available_cols].fillna("Unknown")

        # ✅ Step 5: Convert to JSON
        report_records = display_df.to_dict(orient="records")
        label = gene or drug or "All_Genes"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = str(OUTPUTS_DIR / f"reports/{label}_{timestamp}.pdf").replace("\\", "/")

        # ✅ Step 6: Save in PostgreSQL
        db = SessionLocal()
        try:
            patient = db.query(Patient).filter(Patient.patient_code == "PT-0001").first()
            new_report = Report(
                patient_id=patient.id if patient else None,
                report_title=f"Pharmacogenomics Report - {label}",
                pdf_path=pdf_path,
                report_json=report_records,
                generated_at=datetime.now(),
            )
            db.add(new_report)
            db.commit()
            db.refresh(new_report)
            print(f"✅ Report saved in DB (ID={new_report.id})")
        except Exception as db_err:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {db_err}")
        finally:
            db.close()

        # ✅ Step 7: Return to frontend
        return {
            "message": "Report generated successfully ✅",
            "search_label": label,
            "total_records": len(display_df),
            "columns": available_cols,
            "records": report_records,
            "report_id": new_report.id,
            "report_title": new_report.report_title,
            "pdf_path": pdf_path,
            "patient_code": patient.patient_code if patient else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("❌ Error in /run-report:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate-pdf")
async def generate_pdf(request: Request):
    data = await request.json()
    report_id = data.get("report_id")

    db: Session = SessionLocal()
    try:
        # 🧩 Step 1: Fetch report
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            return {"error": f"Report ID {report_id} not found."}

        # 🧩 Step 2: Fetch patient
        patient = db.query(Patient).filter(Patient.id == report.patient_id).first() if report.patient_id else None

        # 🧩 Step 3: Prepare PDF path
        pdf_path = report.pdf_path or os.path.join(
            "E:/genomics_project/outputs/reports",
            f"{report.report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        # 🧩 Step 4: Create PDF
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        y = height - 50

        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, report.report_title or "Pharmacogenomics Report")
        y -= 30
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 20
        c.line(40, y, width - 40, y)
        y -= 20

        # 🧬 Patient Info
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "🧬 Patient Information")
        y -= 20
        c.setFont("Helvetica", 10)
        full_name = getattr(patient, "full_name", None) or f"{getattr(patient, 'first_name', '')} {getattr(patient, 'last_name', '')}".strip()
        patient_info = [
            f"Name: {full_name or 'Unknown'}",
            f"Patient ID: {getattr(patient, 'patient_code', 'N/A')}",
            f"Age: {getattr(patient, 'age', 'N/A')}",
            f"Gender: {getattr(patient, 'sex', 'Not specified')}",
            f"Clinician: {getattr(patient, 'clinician_name', 'N/A')}"
        ]
        for line in patient_info:
            c.drawString(40, y, line)
            y -= 14

        # 🧩 Report Summary
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Pharmacogenomic Findings")
        y -= 20
        c.setFont("Helvetica", 9)
        report_data = report.report_json if isinstance(report.report_json, list) else json.loads(report.report_json)
        if len(report_data) == 0:
            c.drawString(40, y, "No records available.")
        else:
            for record in report_data[:30]:
                line = ", ".join(f"{k}: {v}" for k, v in record.items() if v and len(str(v)) < 50)
                if y < 60:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 9)
                c.drawString(40, y, line[:110])
                y -= 14

        c.save()

        # 🧩 Step 5: Update DB
        report.pdf_path = pdf_path
        db.commit()
        db.refresh(report)

        print(f"✅ PDF generated successfully for report ID {report.id}")
        return {
            "message": "PDF generated successfully ✅",
            "report_id": report.id,
            "pdf_path": pdf_path
        }

    except Exception as e:
        db.rollback()
        print(f"❌ Error generating PDF: {e}")
        return {"error": f"PDF generation failed: {str(e)}"}
    finally:
        db.close()


@app.post("/save-report")
async def save_report(request: Request):
    """
    Save generated report metadata and file path to PostgreSQL.
    """
    db = SessionLocal()
    try:
        data = await request.json()  # ✅ Use await to correctly parse JSON body

        report = Report(
            report_title=data.get("report_title", "Pharmacogenomics Report"),
            pdf_path=data.get("pdf_path"),
            report_json=json.dumps(data.get("report_json", {})),
            generated_at=datetime.now(),
            patient_id=data.get("patient_id")  # ✅ Optional patient link
        )

        db.add(report)
        db.commit()
        db.refresh(report)

        print(f"✅ Report saved successfully (ID={report.id})")
        return {"message": "Report saved successfully", "report_id": report.id}

    except Exception as e:
        db.rollback()
        print("❌ Error saving report:", e)
        return {"error": str(e)}

    finally:
        db.close()



@app.get("/download-pdf/{report_id}")
def download_pdf(report_id: int):
    db = SessionLocal()
    report = db.query(Report).filter(Report.id == report_id).first()
    db.close()

    if not report or not os.path.exists(report.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        path=report.pdf_path,
        filename=os.path.basename(report.pdf_path),
        media_type="application/pdf"
    )

@app.get("/")
def home():
    return {"message": "Genomic Data Analysis Backend Running ✅"}

app.include_router(auth_router)