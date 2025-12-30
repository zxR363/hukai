import os
import sys

# Set Mock environment variable
os.environ["USE_MOCK"] = "true"

# Add the project root (hukuk) to sys.path
# This ensures that 'from web.mock_legal_engine import ...' works
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: Working directory: {os.getcwd()}")
print(f"DEBUG: Project root: {project_root}")

try:
    # We want to test the logic in main.py without actually loading it (due to dependencies like fastapi)
    # So we simulate the main.py logic here or check the mock file directly
    
    from web.mock_legal_engine import LegalSearchEngine as MockEngine
    
    engine = MockEngine()
    candidates = engine.retrieve_raw_candidates("test query")
    
    print("\n--- Mock Engine Verification ---")
    if len(candidates) > 0 and candidates[0].__class__.__name__ == 'MockHit':
        print("SUCCESS: Mock engine classes are working as expected.")
        print(f"Found {len(candidates)} mock candidates.")
        for c in candidates:
            print(f"- {c.payload['source']} (Score: {c.score})")
    else:
        print("FAILURE: Mock engine is not returning expected objects.")
        
except Exception as e:
    print(f"ERROR during verification: {e}")
