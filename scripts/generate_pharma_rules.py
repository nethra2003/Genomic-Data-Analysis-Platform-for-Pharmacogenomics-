import pandas as pd
import random

# === Input & Output Paths ===
GENE_FILE = r"E:\genomics_project\outputs\unique_genes.txt"
OUTPUT_FILE = r"E:\genomics_project\data\pharma_rules.csv"

# === Load unique gene names ===
with open(GENE_FILE, "r") as f:
    genes = [line.strip() for line in f if line.strip() and line.strip().lower() != "unknown"]

print(f"✅ Loaded {len(genes)} unique genes from {GENE_FILE}")

# === Data pools for randomized realistic rule creation ===
variants = [f"rs{random.randint(100000,999999)}" for _ in range(2000)]
drugs = [
    "Warfarin", "Codeine", "Simvastatin", "Omeprazole", "Clopidogrel", "Tacrolimus",
    "Fluoxetine", "Carbamazepine", "Phenytoin", "Tamoxifen", "Capecitabine",
    "Isoniazid", "Losartan", "Atorvastatin", "Metformin", "Ibuprofen", "Sertraline"
]
conditions = [
    "Hypertension", "Fever", "Cough", "Pain", "Anxiety", "Infection", "Depression",
    "Inflammation", "Allergy", "Asthma", "Cancer", "Fatigue", "Epilepsy", "Diabetes"
]
dosages = ["Increase", "Reduce", "Maintain", "Avoid", "Monitor closely"]
responses = [
    "Poor metabolizer", "Rapid metabolizer", "Intermediate metabolizer",
    "Extensive metabolizer", "Non-expressor", "Sensitive", "Low transporter activity"
]
recommendations = [
    "Use lower dose", "Use higher dose", "Avoid or switch drug", "Monitor for toxicity",
    "Consider alternative therapy", "Reduce dose by 25%", "Adjust based on response"
]

# === Build pharma rules ===
rules = []
for gene in genes:
    rule = [
        gene,
        random.choice(variants),
        random.choice(drugs),
        random.choice(conditions),
        random.choice(dosages),
        random.choice(responses),
        random.choice(recommendations)
    ]
    rules.append(rule)

# === Create DataFrame & Save ===
df_rules = pd.DataFrame(
    rules,
    columns=["Gene", "Variant", "Drug", "Condition", "Dosage", "Response_Type", "Recommendation"]
)
df_rules.to_csv(OUTPUT_FILE, index=False)

print(f"💾 pharma_rules.csv created successfully at: {OUTPUT_FILE}")
print(f"📊 Total rules: {len(df_rules)}")
print(df_rules.head(10))
