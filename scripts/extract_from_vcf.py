import pandas as pd
import gzip
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path("E:/genomics_project")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "extracted"
OUT_DIR.mkdir(exist_ok=True)

# --- Helper function ---
def extract_vcf_data(vcf_path):
    records = []
    opener = gzip.open if vcf_path.suffix == ".gz" else open

    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().split('\t')
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info = parts[:8]

            # Extract useful fields from INFO (Gene, Consequence, Impact)
            gene, consequence, impact = "Unknown", "Unknown", "Unknown"
            if "CSQ=" in info:
                info_part = info.split("CSQ=")[-1].split(",")[0]
                fields = info_part.split("|")
                if len(fields) >= 4:
                    consequence = fields[1]
                    impact = fields[2]
                    gene = fields[3]

            records.append([chrom, pos, vid, ref, alt, qual, gene, consequence, impact])

    return pd.DataFrame(records, columns=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "Gene", "Consequence", "IMPACT"])


# --- Extract all annotated VCFs ---
for vcf_file in DATA_DIR.glob("ALL.chr*_annotated_GRCh37.vcf"):
    print(f"🔍 Processing {vcf_file.name}...")
    df = extract_vcf_data(vcf_file)
    out_file = OUT_DIR / (vcf_file.stem + "_extracted.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved extracted CSV: {out_file}")
