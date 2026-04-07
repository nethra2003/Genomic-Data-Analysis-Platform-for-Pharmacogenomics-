import pandas as pd
from pathlib import Path

def vcf_to_csv(vcf_path, csv_path):
    rows = []
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            chrom, pos, id_, ref, alt, qual, filt, info = parts[:8]
            info_dict = {}
            for item in info.split(";"):
                if "=" in item:
                    key, val = item.split("=", 1)
                    info_dict[key] = val
            rows.append({
                "CHROM": chrom,
                "POS": pos,
                "ID": id_,
                "REF": ref,
                "ALT": alt,
                "QUAL": qual,
                "Gene": info_dict.get("Gene", "Unknown"),
                "Consequence": info_dict.get("Consequence", "Unknown"),
                "IMPACT": info_dict.get("IMPACT", "Unknown"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path} with {len(df)} rows")

# Example usage
vcf_to_csv(
    Path("E:/genomics_project/data/ALL.chr22_annotated_GRCh37.vcf"),
    Path("E:/genomics_project/data/ALL.chr22_annotated.csv")
)
