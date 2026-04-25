import pandas as pd
import os

# Original directory
source_dir = "/home/rpetcu/projects/BackpropTrace/data/nq_hotpotqa_search_decomposed"

# Parallel directory name
target_dir = "/home/rpetcu/projects/BackpropTrace/data/subset_nq_hotpotqa_search_decomposed"

# Ensure the new directory exists
os.makedirs(target_dir, exist_ok=True)

def process_file(name, n):
    src = os.path.join(source_dir, name)
    dst = os.path.join(target_dir, name)
    
    if os.path.exists(src):
        print(f"Processing {name}...")
        df = pd.read_parquet(src)
        # Using head(n) for a fast, deterministic comparison
        subset = df.head(n)
        subset.to_parquet(dst)
        print(f"Successfully saved {len(subset)} rows to {dst}")
    else:
        print(f"Error: {src} not found!")

# Execute for both files
process_file("train.parquet", 1000)
process_file("val.parquet", 32)