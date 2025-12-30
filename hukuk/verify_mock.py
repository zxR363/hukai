import os
import sys

# Set Mock environment variable
os.environ["USE_MOCK"] = "true"

# Add current directory to path
sys.path.append(os.getcwd())

try:
    # Importing main.py should trigger the mock import
    import web.main as main
    
    print("\n--- Verification Test ---")
    print(f"USE_MOCK is set to: {main.USE_MOCK}")
    
    # Check if the classes are indeed from mock_legal_engine
    # In mock_legal_engine, retrieve_raw_candidates returns MockHit objects
    engine = main.LegalSearchEngine()
    candidates = engine.retrieve_raw_candidates("test query")
    
    if len(candidates) > 0 and candidates[0].__class__.__name__ == 'MockHit':
        print("✅ SUCCESS: main.py is using the Mock engine.")
    else:
        print("❌ FAILURE: main.py is NOT using the Mock engine correctly.")
        
except Exception as e:
    print(f"❌ ERROR during verification: {e}")
