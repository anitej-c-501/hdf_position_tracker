import os
import h5py

def validate_folder(folder_path, create_if_missing=False):
    if not os.path.exists(folder_path):
        if create_if_missing:
            os.makedirs(folder_path)
        else:
            raise ValueError(f"Folder '{folder_path}' does not exist.")

def list_hdf5_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".hdf5")]
    if not files:
        raise ValueError(f"No valid HDF5 files found in folder: {folder_path}")
    return files

def open_hdf5_file(file_path):
    try:
        return h5py.File(file_path, "r")
    except Exception as e:
        raise ValueError(f"Failed to open HDF5 file '{file_path}': {e}")