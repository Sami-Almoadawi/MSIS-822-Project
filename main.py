import sys
import os

# Add 'src' directory to the system path to allow importing modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import the modeling module from 'src'
    import modeling 
except ImportError as e:
    print(f"Error importing modules: {e}")
    
    print("Ensure 'modeling.py' is inside the 'src' folder.")
    sys.exit(1)

def main():
    print("🚀 Starting the Project Pipeline...")
    print("="*40)

    # PHASE 4: MACHINE LEARNING CLASSIFICATION MODELS
    
    print("\n[Phase 4] Starting Model Training & Evaluation...")
    
    # Check for 'main' function in the imported module and execute it
    if hasattr(modeling, 'main'):
        modeling.main()
    elif hasattr(modeling, 'train'):
        modeling.train()
    else:
        print("Model script executed successfully upon import.")

    print("\n" + "="*40)
    print("✅ Process Finished Successfully!")

if __name__ == "__main__":
    main()
