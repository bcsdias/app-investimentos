import os
import sys

# Add src to path
BASE_DIR = r"c:\onedrive-bcsdias\OneDrive\dev\app_investimentos"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def verify_paths():
    print("Verifying directory paths calculated by the system...")
    
    from src.data.sources.market_data import DATA_RAW_DIR, DATA_DOWNLOADS_DIR, BASE_DIR as MD_BASE_DIR
    from src.data.sources.b3_source import BASE_DIR as B3_BASE_DIR, DATA_STATIC_DIR
    from src.engine.financial_report import BASE_DIR as FR_BASE_DIR
    
    expected_root = r"c:\onedrive-bcsdias\OneDrive\dev\app_investimentos"
    
    print(f"Market Data BASE_DIR: {MD_BASE_DIR}")
    print(f"B3 Source BASE_DIR: {B3_BASE_DIR}")
    print(f"Financial Report BASE_DIR: {FR_BASE_DIR}")
    
    assert MD_BASE_DIR.lower() == expected_root.lower()
    assert B3_BASE_DIR.lower() == expected_root.lower()
    assert FR_BASE_DIR.lower() == expected_root.lower()
    
    print(f"DATA_RAW_DIR: {DATA_RAW_DIR}")
    print(f"DATA_DOWNLOADS_DIR: {DATA_DOWNLOADS_DIR}")
    print(f"DATA_STATIC_DIR: {DATA_STATIC_DIR}")
    
    # Check if directories exist
    dirs = [DATA_RAW_DIR, DATA_DOWNLOADS_DIR, DATA_STATIC_DIR]
    for d in dirs:
        if os.path.exists(d):
            print(f"EXISTS: {d}")
        else:
            print(f"MISSING: {d}")
            # The code should have created them if called
            # But here we just import, so they might not be created if they weren't there
            # Actually, market_data.py has os.makedirs in the module level now.
            if os.path.exists(d):
                print(f"CREATED: {d}")

    print("Path verification complete!")

if __name__ == "__main__":
    verify_paths()
