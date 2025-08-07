import sys
import os

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(script_path)

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Insert the parent directory into the system path
sys.path.insert(0, parent_dir)