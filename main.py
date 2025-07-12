import subprocess
import sys
import os

def run_script(script_name):
    """
    Runs a Python script using subprocess.
    """
    print(f"--- Running {script_name} ---")
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True, check=True)
        print(f"Output from {script_name}:\n{result.stdout}")
        if result.stderr:
            print(f"Errors from {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: {script_name} not found. Ensure it's in the same directory as main.py or provide the full path.")
    print(f"--- Finished {script_name} ---")
    print("\n")

if __name__ == "__main__":
    # Ensure the current working directory is correct if scripts expect specific relative paths
    # For example, if your scripts are in a 'scripts' subdirectory and you run main.py from the parent directory,
    # you might need to adjust paths within the individual scripts or change directory here.
    # e.g., os.chdir('scripts') before running, and os.chdir('..') afterwards.

    # Run outreacher.py
    run_script("/files/outreacher_role/outreacher.py")

    # Run cw.py (content writer)
    run_script("/files/content_writer/cw.py")

    # Run pr.py (prospector)
    run_script("/files/prospector/pr.py")

    print("All scripts have been executed.")