import pandas as pd
from pathlib import Path

# Define paths to your extracted files
files = {
    "chr1": Path("E:/genomics_project/data/extracted/ALL.chr1_annotated_GRCh37_extracted.csv"),
    "chr3": Path("E:/genomics_project/data/extracted/ALL.chr3_annotated_GRCh37_extracted.csv"),
    "chr22": Path("E:/genomics_project/data/extracted/ALL.chr22_annotated_GRCh37_extracted.csv"),
}

output_path = Path("E:/genomics_project/outputs/gene_summary.csv")

all_genes = set()
summary_records = []

print("🧬 Scanning annotated chromosome files...\n")

for chr_name, file_path in files.items():
    if not file_path.exists():
        print(f"⚠️ Missing file: {file_path}")
        continue

    print(f"📂 Reading {file_path.name} ...")
    df = pd.read_csv(file_path)

    # Drop missing gene entries and count
    df = df[df["Gene"].notnull() & (df["Gene"] != "")]
    genes = df["Gene"].unique().tolist()
    all_genes.update(genes)

    # Count variants per gene
    gene_counts = df["Gene"].value_counts().reset_index()
    gene_counts.columns = ["Gene", "Variant_Count"]
    gene_counts["Chromosome"] = chr_name

    summary_records.append(gene_counts)

# Combine all chromosomes
if summary_records:
    summary_df = pd.concat(summary_records, ignore_index=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\n✅ Gene summary saved to: {output_path}")
else:
    print("❌ No valid gene data found!")

# Display unique gene list
print(f"\n🧬 Total unique genes across all chromosomes: {len(all_genes)}")
print(sorted(list(all_genes))[:50])  # print first 50 genes alphabetically

# Optionally save full list
gene_list_path = Path("E:/genomics_project/outputs/unique_genes.txt")
with open(gene_list_path, "w") as f:
    for gene in sorted(list(all_genes)):
        f.write(gene + "\n")
print(f"🗂️ Unique gene list saved to: {gene_list_path}")
