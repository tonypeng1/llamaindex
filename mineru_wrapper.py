import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def run_mineru(input_pdf, output_dir):
    """
    Runs MinerU CLI to parse a PDF.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the command
    # Using vlm-mlx-engine for macOS as it's faster
    # If it fails, we could fallback to 'pipeline'
    cmd = [
        "./.mineru_env/bin/mineru",
        "-p", input_pdf,
        "-o", output_dir,
        "-b", "vlm-mlx-engine"
    ]
    
    print(f"🚀 Running MinerU: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ MinerU finished successfully.")
        # print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ MinerU failed with error:\n{e.stderr}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mineru_wrapper.py <input_pdf> <output_dir>")
        sys.exit(1)
        
    input_pdf = sys.argv[1]
    output_dir = sys.argv[2]
    
    success = run_mineru(input_pdf, output_dir)
    if not success:
        sys.exit(1)
