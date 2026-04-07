# import pandas as pd
# from config import CHR_PATHS, RULES_FILE, DATA_DIR, OUTPUT_DIR
# from pathlib import Path

# def load_variant_data():
#     dfs = []
#     for chr_name, path in CHR_PATHS.items():
#         print(f"Loading {chr_name} from {path} ...")
#         df = pd.read_csv(path)
#         df["chromosome"] = chr_name
#         dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)

# def build_training_dataset():
#     variants = load_variant_data()
#     rules = pd.read_csv(RULES_FILE)

#     # Merge variants with pharmacogenomic rules
#     merged = pd.merge(
#         variants,
#         rules,
#         how="left",
#         left_on=["rsid", "gene"],
#         right_on=["variant_id", "gene"]
#     )

#     # Drop columns we don’t need
#     merged = merged.drop(columns=["variant_id"], errors="ignore")

#     # Save processed dataset
#     output_path = OUTPUT_DIR / "training_dataset.csv"
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     merged.to_csv(output_path, index=False)
#     print(f"✅ Training dataset saved to {output_path}")

# if __name__ == "__main__":
#     build_training_dataset()

# import pandas as pd
# from config import CHR_PATHS, RULES_FILE, OUTPUT_DIR
# from pathlib import Path

# def load_variant_data():
#     dfs = []
#     for chr_name, path in CHR_PATHS.items():
#         print(f"📂 Loading {chr_name} from {path} ...")
#         df = pd.read_csv(path)
#         df["chromosome"] = chr_name
#         dfs.append(df)
#     merged_df = pd.concat(dfs, ignore_index=True)
#     print(f"✅ Combined {len(dfs)} chromosomes → {merged_df.shape[0]} total variants.")
#     return merged_df

# def clean_variant_data(df):
#     # Normalize common column names
#     df.columns = df.columns.str.lower()
#     possible_cols = ["rsid", "variant_id", "gene", "genotype", "consequence", "impact", "sample_id"]
#     for col in possible_cols:
#         if col not in df.columns:
#             df[col] = None
#     return df

# def merge_with_rules(variants, rules):
#     merged = pd.merge(
#         variants,
#         rules,
#         how="left",
#         left_on=["rsid", "gene"],
#         right_on=["variant_id", "gene"]
#     )
#     merged["response"] = merged["response"].fillna("unknown")
#     merged = merged.drop(columns=["variant_id"], errors="ignore")
#     return merged

# def build_training_dataset():
#     variants = load_variant_data()
#     variants = clean_variant_data(variants)
#     rules = pd.read_csv(RULES_FILE)

#     dataset = merge_with_rules(variants, rules)

#     # Convert genotypes to numeric encodings for ML
#     genotype_map = {"0/0": 0, "0/1": 1, "1/1": 2, "./.": -1}
#     if "genotype" in dataset.columns:
#         dataset["genotype_code"] = dataset["genotype"].map(genotype_map).fillna(-1)

#     output_path = OUTPUT_DIR / "training_dataset.csv"
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     dataset.to_csv(output_path, index=False)
#     print(f"💾 Training dataset saved to: {output_path}")
#     print(f"Shape: {dataset.shape}")
#     print(f"Columns: {list(dataset.columns)}")

# if __name__ == "__main__":
#     build_training_dataset()


# 


# import pandas as pd
# from pathlib import Path
# from config import CHR_PATHS, OUTPUT_DIR, RULES_FILE

# def build_training_dataset():
#     combined = []

#     print("📂 Loading chromosome data...")
#     for chr_name, file_path in CHR_PATHS.items():
#         if Path(file_path).exists():
#             print(f"  ➜ Loading {chr_name} from {file_path}")
#             df = pd.read_csv(file_path)
#             df["chromosome"] = chr_name
#             combined.append(df)
#         else:
#             print(f"⚠️ File missing: {file_path}")

#     if not combined:
#         raise FileNotFoundError("❌ No chromosome files found in data directory!")

#     dataset = pd.concat(combined, ignore_index=True)
#     print(f"✅ Combined {len(dataset)} total variants.")

#     # Clean and prepare variant IDs
#     dataset = dataset[dataset["ID"].notna()]          # Drop missing IDs
#     dataset["ID"] = dataset["ID"].astype(str)
#     dataset = dataset[dataset["ID"].str.startswith("rs", na=False)]  # Keep rsIDs only

#     print(f"🧬 Filtered to {len(dataset)} variants with valid rsIDs.")

#     # Load pharma rules
#     rules = pd.read_csv(RULES_FILE)
#     print(f"📖 Loaded {len(rules)} pharma rules.")

#     # Merge variant data with rules
#     merged = dataset.merge(
#         rules,
#         left_on="ID",
#         right_on="Variant",
#         how="left"
#     )

#     print(f"🔗 Merged dataset shape: {merged.shape}")

#     # Save merged dataset
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     output_path = OUTPUT_DIR / "training_dataset.csv"
#     merged.to_csv(output_path, index=False)
#     print(f"💾 Saved training dataset to: {output_path}")

#     # Small preview
#     print("\n🔎 Preview of merged data:")
#     print(merged.head(10))

# if __name__ == "__main__":
#     build_training_dataset()

# import pandas as pd
# from pathlib import Path
# from config import CHR_PATHS, OUTPUT_DIR, RULES_FILE

# def build_training_dataset():
#     print("📖 Loading pharmacogenomic rules...")
#     rules = pd.read_csv(RULES_FILE)
#     rules["Gene"] = rules["Gene"].astype(str).str.upper()
#     print(f"✅ Loaded {len(rules)} pharma rules.")

#     OUTPUT_DIR.mkdir(exist_ok=True)
#     output_path = OUTPUT_DIR / "training_dataset.csv"

#     # Write headers only once
#     wrote_header = False

#     for chr_name, file_path in CHR_PATHS.items():
#         if not Path(file_path).exists():
#             print(f"⚠️ Missing file: {file_path}")
#             continue

#         print(f"📂 Processing {chr_name} in chunks from {file_path}...")
#         chunk_iter = pd.read_csv(file_path, chunksize=200000, low_memory=False)

#         for i, chunk in enumerate(chunk_iter):
#             chunk["chromosome"] = chr_name

#             # Ensure Gene column exists (placeholder)
#             if "Gene" not in chunk.columns:
#                 chunk["Gene"] = "UNKNOWN"

#             # Standardize case
#             chunk["Gene"] = chunk["Gene"].astype(str).str.upper()

#             # Merge small rules table (only 99 rows)
#             merged_chunk = chunk.merge(rules, on="Gene", how="left")

#             # Append chunk directly to output file
#             merged_chunk.to_csv(output_path, mode="a", index=False, header=not wrote_header)
#             wrote_header = True

#             print(f"   ➜ Processed chunk {i+1} ({len(merged_chunk)} rows)")

#     print(f"💾 Training dataset saved at: {output_path}")

# if __name__ == "__main__":
#     build_training_dataset()

# import pandas as pd
# from config import CHR_PATHS, OUTPUT_DIR, RULES_FILE
# from pathlib import Path

# def build_training_dataset():
#     combined = []

#     print("📂 Loading annotated chromosome data...")
#     for chr_name, file_path in CHR_PATHS.items():
#         if Path(file_path).exists():
#             print(f"  ➜ Loading {chr_name} from {file_path}")
#             df = pd.read_csv(file_path)
#             df["chromosome"] = chr_name
#             combined.append(df)
#         else:
#             print(f"⚠️ Missing file: {file_path}")

#     if not combined:
#         raise FileNotFoundError("❌ No annotated chromosome files found!")

#     dataset = pd.concat(combined, ignore_index=True)
#     print(f"✅ Combined {len(dataset)} variants from all chromosomes.")

#     # Remove entries with missing Gene info
#     dataset = dataset.dropna(subset=["Gene"])
#     dataset = dataset[dataset["Gene"].str.strip() != ""]

#     # Load pharmacogenomic rules
#     print(f"📖 Loading pharmacogenomic rules from {RULES_FILE} ...")
#     rules = pd.read_csv(RULES_FILE)
#     print(f"✅ Loaded {len(rules)} rules.")

#     # Merge by gene name
#     merged = pd.merge(dataset, rules, on="Gene", how="left")

#     print(f"🔗 Merged dataset shape: {merged.shape}")

#     # Save merged dataset
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     out_file = OUTPUT_DIR / "training_dataset.csv"
#     merged.to_csv(out_file, index=False)

#     print(f"💾 Saved final training dataset to: {out_file}")

#     print("\n🔎 Preview of merged data:")
#     print(merged.head(20))

# if __name__ == "__main__":
#     build_training_dataset()



# 




# 


import pandas as pd
from pathlib import Path
from config import OUTPUT_DIR, RULES_FILE

ANNOTATED_FILES = [
    r"E:\genomics_project\data\extracted\ALL.chr1_annotated_GRCh37_extracted.csv",
    r"E:\genomics_project\data\extracted\ALL.chr3_annotated_GRCh37_extracted.csv",
    r"E:\genomics_project\data\extracted\ALL.chr22_annotated_GRCh37_extracted.csv"
]

def build_training_dataset():
    print("📖 Loading pharmacogenomic rules...")
    rules = pd.read_csv(RULES_FILE)
    rules["Gene"] = rules["Gene"].astype(str).str.strip().str.upper()
    print(f"✅ Loaded {len(rules)} rules.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "training_dataset.csv"

    # Write header once
    pd.DataFrame(columns=[
        "CHROM","POS","ID","REF","ALT","QUAL","Gene","Consequence","IMPACT",
        "Variant","Drug","Condition","Dosage","Response_Type","Recommendation"
    ]).to_csv(output_path, index=False)

    # Process each annotated chromosome file in chunks
    for file_path in ANNOTATED_FILES:
        print(f"\n📂 Processing {file_path}...")
        if not Path(file_path).exists():
            print(f"⚠️ Missing: {file_path}")
            continue

        # Read in 100k-line chunks to stay memory-safe
        chunks = pd.read_csv(file_path, chunksize=100_000)
        for i, chunk in enumerate(chunks):
            chunk["Gene"] = chunk["Gene"].astype(str).str.strip().str.upper()
            merged = chunk.merge(rules, on="Gene", how="left")

            merged.to_csv(output_path, mode="a", index=False, header=False)
            if i % 10 == 0:
                print(f"  ✔️ processed {i*100_000:,} rows...")

    print(f"\n💾 Final training dataset saved to: {output_path}")

if __name__ == "__main__":
    build_training_dataset()
