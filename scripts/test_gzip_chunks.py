import gzip

filename = r"E:\genomics_data\raw_vcf\ALL.chr1.vcf.gz"
chunk_size = 1024 * 1024 * 10  # 10 MB

try:
    with gzip.open(filename, "rb") as f:
        while f.read(chunk_size):
            pass
    print("Gzip file is OK — no corruption detected.")
except Exception as e:
    print("Gzip file error:", e)