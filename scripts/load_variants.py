import psycopg2
import pandas as pd
import glob

# Database connection settings
DB_SETTINGS = {
    "dbname": "pharmacogenomics",
    "user": "pharma_user",
    "password": "Pharma@123",
    "host": "localhost",
    "port": "5432"
}

# Connect to PostgreSQL
conn = psycopg2.connect(**DB_SETTINGS)
cur = conn.cursor()

# Path to processed variant CSVs
csv_files = glob.glob(r"E:\genomics_data\processed\ALL.chr*_variants.csv")

for file in csv_files:
    print(f"Loading {file} ...")
    df = pd.read_csv(file)
    
    # Insert rows into the database
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO variants (chrom, pos, ref, alt, qual, filter, info)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (
            row['CHROM'], row['POS'], row['REF'], row['ALT'],
            row.get('QUAL'), row.get('FILTER'), row.get('INFO')
        ))

    conn.commit()
    print(f"Loaded {len(df)} rows from {file}")

cur.close()
conn.close()
print("=== All variant data loaded successfully! ===")
