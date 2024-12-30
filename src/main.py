import argparse
import os
from processor import process_data

def resolve_path(path):
    """
    Resolve a relative or absolute path to its absolute path.
    :param path: Input path (relative or absolute)
    :return: Absolute path
    """
    return os.path.abspath(path)

def main():
    # Create argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Process and aggregate HDF5 tracking data.")
    parser.add_argument("input_folder", help="Path to the folder containing input HDF5 files.")
    parser.add_argument("output_folder", help="Path to the folder where output CSV files will be saved.")

    # Parse the arguments
    args = parser.parse_args()

    # Resolve paths to absolute paths
    input_folder = resolve_path(args.input_folder)
    output_folder = resolve_path(args.output_folder)

    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")

    # Process the data
    try:
        process_data(input_folder, output_folder)
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
