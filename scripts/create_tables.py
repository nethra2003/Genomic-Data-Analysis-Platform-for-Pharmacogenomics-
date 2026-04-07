import psycopg2

# Database connection details
DB_NAME = "pharmacogenomics"
DB_USER = "pharma_user"
DB_PASSWORD = "Pharma@123"
DB_HOST = "localhost"
DB_PORT = "5432"

# SQL commands to create tables
commands = [
    """
    CREATE TABLE IF NOT EXISTS variants (
        id SERIAL PRIMARY KEY,
        chrom VARCHAR(10),
        pos BIGINT,
        variant_id VARCHAR(100),
        ref VARCHAR(500),
        alt VARCHAR(500),
        qual FLOAT,
        source_file VARCHAR(100)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS annotations (
        id SERIAL PRIMARY KEY,
        variant_id VARCHAR(100),
        gene_name VARCHAR(100),
        effect VARCHAR(200),
        clinical_significance VARCHAR(200)
    );
    """
]

try:
    # Connect to the PostgreSQL server
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    # Execute each table creation command
    for command in commands:
        cur.execute(command)

    conn.commit()
    print("Tables created successfully!")

    cur.close()
    conn.close()

except Exception as e:
    print(" Error creating tables:", e)
