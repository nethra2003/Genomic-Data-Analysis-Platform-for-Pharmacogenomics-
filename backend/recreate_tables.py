from db_config import Base, engine
import importlib
from sqlalchemy import text

print("⚙️ Forcefully recreating all tables in the pharmacogenomics database...")

# Dynamically import all model definitions from models.py
try:
    models = importlib.import_module("models")
    print("✅ Models imported successfully.")
except Exception as e:
    print(f"❌ Error importing models: {e}")

# Drop all tables using CASCADE
with engine.connect() as connection:
    try:
        print("⚠️ Dropping all existing tables (CASCADE)...")
        connection.execute(text("DROP SCHEMA public CASCADE;"))
        connection.execute(text("CREATE SCHEMA public;"))
        print("✅ All existing tables dropped successfully.")
    except Exception as e:
        print(f"❌ Error dropping schema: {e}")

# Recreate all tables from models
try:
    Base.metadata.create_all(bind=engine)
    print("✅ All tables recreated successfully in the pharmacogenomics database.")
except Exception as e:
    print(f"❌ Error recreating tables: {e}")

