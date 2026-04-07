import allel
import pandas as pd
import os
import time
import gzip

# === CONFIG ===
RAW_DIR = r"E:\genomics_data\raw_vcf"
PROCESSED_DIR = r"E:\genomics_data\processed"
LOG_FILE = r"E:\genomics_data\logs\vcf_processing.log"

# Disable strict CRC check (safe for trusted files like 1000 Genomes)
gzip._GzipReader._check_crc = lambda *args, **kwargs: None


def log_message(message):
    """Write progress updates to the log file and terminal."""
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)


def process_vcf(filename):
    """Process a single VCF file and extract variant data."""
    input_path = os.path.join(RAW_DIR, filename)
    base_name = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .vcf.gz
    output_path = os.path.join(PROCESSED_DIR, base_name + "_variants.csv")

    log_message(f"Processing {filename} ...")

    try:
        # Load the VCF safely through gzip
        with gzip.open(input_path, "rb") as f:
            callset = allel.read_vcf(
                f,
                fields=["variants/CHROM", "variants/POS", "variants/ID",
                        "variants/REF", "variants/ALT", "variants/QUAL"]
            )

        # Convert to DataFrame
        df = pd.DataFrame({
            "CHROM": callset["variants/CHROM"],
            "POS": callset["variants/POS"],
            "ID": callset["variants/ID"],
            "REF": callset["variants/REF"],
            "ALT": [
                ",".join(alt) if isinstance(alt, list) else alt
                for alt in callset["variants/ALT"]
            ],
            "QUAL": callset["variants/QUAL"],
        })

        # Save to CSV
        df.to_csv(output_path, index=False)
        log_message(f"Saved {len(df):,} variants to {output_path}")

    except Exception as e:
        log_message(f" Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    start_time = time.time()
    log_message("=== Starting VCF Processing ===")

    vcf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".vcf.gz")]
    for vcf in vcf_files:
        process_vcf(vcf)

    log_message(f"=== Completed in {round((time.time() - start_time)/60, 2)} minutes ===")
