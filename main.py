import subprocess
import sys

def run_module(module_name):
    """
    Runs a Python module and handles any errors.
    """
    try:
        print(f"Running {module_name}...")
        subprocess.run([sys.executable, "-m", f"src.{module_name}"], check=True)
        print(f"{module_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {module_name}: {e}\n")
        sys.exit(1)

def main():
    """
    Main function to run data preprocessing, training, and inference modules sequentially.
    """
    print("Starting the workflow...\n")

    # Step 1: Run data preprocessing
    run_module("data")

    # Step 2: Train the model
    run_module("train")

    # Step 3: Evaluate the model
    run_module("eval")

    # Step 4: Run inference
    run_module("infer")

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()