import sys
from pathlib import Path

# Reconfigure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def find_in_file():
    filepath = Path("data/raw_docs/Test CSR 21_26 .txt")
    if not filepath.exists():
        print("File not found.")
        return
    
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    
    search_terms = ["jhawar", "manoj"]
    for term in search_terms:
        print(f"\nSearching for '{term}':")
        found = False
        for idx, line in enumerate(lines):
            if term in line.lower():
                print(f"Line {idx+1}: {line.strip()}")
                found = True
        if not found:
            print("No matches found.")

if __name__ == "__main__":
    find_in_file()
