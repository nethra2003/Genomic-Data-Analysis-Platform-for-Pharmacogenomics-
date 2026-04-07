import psycopg2

# Database connection details
DB_NAME = "pharmacogenomics"
DB_USER = "pharma_user"
DB_PASSWORD = "Pharma@123"  # change if you used a different password
DB_HOST = "localhost"
DB_PORT = "5432"

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(" Connection failed:", e)
