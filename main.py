import sys
import os

# Add 'src' directory to the system path to allow importing modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import the training module from 'src'
    import train_model 
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure 'train_model.py' is inside the 'src' folder.")
    sys.exit(1)

def main():
    print("🚀 Starting the Project Pipeline...")
    print("="*40)

    # --- Phase 4: Model Training ---
    print("\n[Phase 4] Starting Model Training & Evaluation...")
    
    # Check for 'main' or 'train' function in the imported module and execute it
    if hasattr(train_model, 'main'):
        train_model.main()
    elif hasattr(train_model, 'train'):
        train_model.train()
    else:
        print("Model script executed successfully upon import.")

    print("\n" + "="*40)
    print("✅ Process Finished Successfully!")

if __name__ == "__main__":
    main()