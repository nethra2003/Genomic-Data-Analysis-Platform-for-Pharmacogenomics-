from pathlib import Path

# Base directories
BASE_DIR = Path("E:/genomics_project")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SCRIPT_DIR = BASE_DIR / "scripts"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Chromosome CSV files
CHR_PATHS = {
    "chr1": DATA_DIR / "extracted/ALL.chr1_annotated_GRCh37_extracted.csv",
    "chr3": DATA_DIR / "extracted/ALL.chr3_annotated_GRCh37_extracted.csv",
    "chr22": DATA_DIR / "extracted/ALL.chr22_annotated_GRCh37_extracted.csv",
}

# Annotated VCFs (for reference, optional)
ANNOTATED_VCF = {
    "chr1": DATA_DIR / "ALL.chr1_annotated_GRCh37.vcf",
    "chr3": DATA_DIR / "ALL.chr3_annotated_GRCh37.vcf",
    "chr22": DATA_DIR / "ALL.chr22_annotated_GRCh37.vcf",
}

# Rules file for pharmacogenomic mapping (you will create next)
RULES_FILE = DATA_DIR / "pharma_rules.csv"
