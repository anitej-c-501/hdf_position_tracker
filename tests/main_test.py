import csv
import re
import shutil
import tempfile
import h5py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from file_handler import list_hdf5_files, validate_folder
from utils import compute_average_position, compute_max_distance
from processor import process_data, process_hdf5_file
from main import resolve_path

def test_compute_average_position():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])    
    result = compute_average_position(data)   
    assert result == [4.0, 5.0, 6.0], f"Expected [4.0, 5.0, 6.0] but got {result}"

def test_compute_max_distance():
    data = np.array([[3, 4, 0], [6, 8, 0], [0, 0, 0]])    
    result = compute_max_distance(data)    
    assert result == 10.0, f"Expected 10.0 but got {result}"

def test_hdf_file_1():
    file_path = "data\\1.hdf5"
    new_file_path = resolve_path(file_path)
    average_positions, max_distances, sensor_identifiers = process_hdf5_file(new_file_path)
    expected_avg_positions = [188.29965209960938, -207.0303192138672, -14.339226722717285]
    expected_max_distances = [280.26392]
    expected_sensor_identifiers = ["Device_792888129_0_Sensor_0"]  
    assert average_positions == expected_avg_positions, f"Expected {expected_avg_positions}, got {average_positions}"
    assert max_distances == expected_max_distances, f"Expected {expected_max_distances}, got {max_distances}"
    assert sensor_identifiers == expected_sensor_identifiers, f"Expected {expected_sensor_identifiers}, got {sensor_identifiers}"

def test_hdf_file_2():
    file_path = "data\\2.hdf5"
    new_file_path = resolve_path(file_path)
    average_positions, max_distances, sensor_identifiers = process_hdf5_file(new_file_path)
    expected_avg_positions = [193.89981079101562,-249.04673767089844,-8.512978553771973]
    expected_max_distances = [315.8103]
    expected_sensor_identifiers = ["Device_792888129_0_Sensor_0"] 
    assert average_positions == expected_avg_positions, f"Expected {expected_avg_positions}, got {average_positions}"
    assert max_distances == expected_max_distances, f"Expected {expected_max_distances}, got {max_distances}"
    assert sensor_identifiers == expected_sensor_identifiers, f"Expected {expected_sensor_identifiers}, got {sensor_identifiers}"

def test_hdf_file_3():
    file_path = "data\\3.hdf5"
    new_file_path = resolve_path(file_path)
    average_positions, max_distances, sensor_identifiers = process_hdf5_file(new_file_path)
    expected_avg_positions = [173.57557678222656,-299.2967834472656,-35.915374755859375]
    expected_max_distances = [347.91016]
    expected_sensor_identifiers = ["Device_792888129_0_Sensor_0"]  
    assert average_positions == expected_avg_positions, f"Expected {expected_avg_positions}, got {average_positions}"
    assert max_distances == expected_max_distances, f"Expected {expected_max_distances}, got {max_distances}"
    assert sensor_identifiers == expected_sensor_identifiers, f"Expected {expected_sensor_identifiers}, got {sensor_identifiers}"

def test_hdf_file_4():
    file_path = "data\\4.hdf5"
    new_file_path = resolve_path(file_path)
    average_positions, max_distances, sensor_identifiers = process_hdf5_file(new_file_path)
    expected_avg_positions = [325.2101135253906,-221.2873992919922,-19.490367889404297]
    expected_max_distances = [393.97943]
    expected_sensor_identifiers = ["Device_792888129_0_Sensor_0"]  
    assert average_positions == expected_avg_positions, f"Expected {expected_avg_positions}, got {average_positions}"
    assert max_distances == expected_max_distances, f"Expected {expected_max_distances}, got {max_distances}"
    assert sensor_identifiers == expected_sensor_identifiers, f"Expected {expected_sensor_identifiers}, got {sensor_identifiers}"

def test_hdf_file_5():
    file_path = "data\\5.hdf5"
    new_file_path = resolve_path(file_path)
    average_positions, max_distances, sensor_identifiers = process_hdf5_file(new_file_path)
    expected_avg_positions = [328.0667724609375,-100.99529266357422,-18.465534210205078]
    expected_max_distances = [343.83514]
    expected_sensor_identifiers = ["Device_792888129_0_Sensor_0"]  
    assert average_positions == expected_avg_positions, f"Expected {expected_avg_positions}, got {average_positions}"
    assert max_distances == expected_max_distances, f"Expected {expected_max_distances}, got {max_distances}"
    assert sensor_identifiers == expected_sensor_identifiers, f"Expected {expected_sensor_identifiers}, got {sensor_identifiers}"

def test_no_valid_in_input_folder():
    folder_name = "empty_test_folder"
    new_folder_path = resolve_path(folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    for file in os.listdir(new_folder_path):
        os.remove(os.path.join(new_folder_path, file))
    with pytest.raises(ValueError, match=rf"No valid HDF5 files found in folder: {re.escape(new_folder_path)}"):
        list_hdf5_files(new_folder_path)


def test_validate_folder():
    # Test case 1: Folder exists
    existing_folder = "test_existing_folder"
    existing_folder = resolve_path(existing_folder)
    os.makedirs(existing_folder, exist_ok=True)
    try:
        validate_folder(existing_folder)
    except ValueError:
        pytest.fail("validate_folder raised ValueError for an existing folder.")

    # Test case 2: Folder does not exist, create_if_missing=False
    non_existing_folder = "test_non_existing_folder"
    if os.path.exists(non_existing_folder):
        os.rmdir(non_existing_folder)  # Ensure folder doesn't exist
    with pytest.raises(ValueError, match=f"Folder '{non_existing_folder}' does not exist."):
        validate_folder(non_existing_folder, create_if_missing=False)

    # Test case 3: Folder does not exist, create_if_missing=True
    try:
        validate_folder(non_existing_folder, create_if_missing=True)
        assert os.path.exists(non_existing_folder), "Folder was not created when create_if_missing=True."
    finally:
        if os.path.exists(non_existing_folder):
            os.rmdir(non_existing_folder)
    if os.path.exists(existing_folder):
        os.rmdir(existing_folder)

def create_mock_hdf5_file(file_path, avg_positions, max_distances, sensor_identifiers):
    """
    Create a mock HDF5 file with the given data, including a 'Position' dataset.
    """
    with h5py.File(file_path, 'w') as hdf:
        hdf.create_dataset("average_positions", data=avg_positions)
        hdf.create_dataset("max_distances", data=max_distances)
        hdf.create_dataset("sensor_identifiers", data=sensor_identifiers)


def read_csv(file_path):
    """
    Read a CSV file and return its content as a list of rows.
    """
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return [row for row in reader]

def test_process_data_without_position():
    temp_dir = tempfile.mkdtemp()
    input_folder = os.path.join(temp_dir, "input")
    output_folder = os.path.join(temp_dir, "output")
    os.makedirs(input_folder)
    
    try:
        # Create mock HDF5 files
        create_mock_hdf5_file(
            os.path.join(input_folder, "file1.hdf5"),
            avg_positions=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            max_distances=[10.0, 20.0],
            sensor_identifiers=["Sensor_0", "Sensor_1"]
        )
        create_mock_hdf5_file(
            os.path.join(input_folder, "file2.hdf5"),
            avg_positions=[[7.0, 8.0, 9.0]],
            max_distances=[30.0],
            sensor_identifiers=["Sensor_0"]
        )
        hdf5_files = list_hdf5_files(input_folder)
        print(hdf5_files)

        # Process data
        process_data(input_folder, output_folder)

        # Verify CSV contents
        avg_positions_csv = os.path.join(output_folder, "average_positions.csv")
        max_distances_csv = os.path.join(output_folder, "max_distances.csv")

        assert os.path.exists(avg_positions_csv), "Average positions CSV file not found."
        assert os.path.exists(max_distances_csv), "Max distances CSV file not found."

        # Debug CSV contents
        with open(avg_positions_csv, "r") as file:
            print("Average Positions CSV:")
            print(file.read())

        with open(max_distances_csv, "r") as file:
            print("Max Distances CSV:")
            print(file.read())

        # Verify CSV data
        avg_positions_data = read_csv(avg_positions_csv)
        expected_avg_positions = [
            ["File Name"]
        ]
        assert avg_positions_data == expected_avg_positions, f"Unexpected average positions data: {avg_positions_data}"

        max_distances_data = read_csv(max_distances_csv)
        expected_max_distances = [
            ["File Name"]
        ]
        assert max_distances_data == expected_max_distances, f"Unexpected max distances data: {max_distances_data}"

    finally:
        shutil.rmtree(temp_dir)

def test_process_data():
    input_folder = resolve_path("data")
    output_folder = resolve_path("output_folder")
    hdf5_files = list_hdf5_files(input_folder)
    print(hdf5_files)

    # Process data
    process_data(input_folder, output_folder)

    # Verify CSV contents
    avg_positions_csv = os.path.join(output_folder, "average_positions.csv")
    max_distances_csv = os.path.join(output_folder, "max_distances.csv")

    assert os.path.exists(avg_positions_csv), "Average positions CSV file not found."
    assert os.path.exists(max_distances_csv), "Max distances CSV file not found."

    # Verify CSV data
    avg_positions_data = read_csv(avg_positions_csv)
    expected_avg_positions = [["File Name", "Device_792888129_0_Sensor_0_X", "Device_792888129_0_Sensor_0_Y", "Device_792888129_0_Sensor_0_Z"],
    ["1.hdf5", '188.29965209960938', '-207.0303192138672', '-14.339226722717285'],
    ["2.hdf5", '193.89981079101562', '-249.04673767089844', '-8.512978553771973'],
    ["3.hdf5", '173.57557678222656', '-299.2967834472656', '-35.915374755859375'],
    ["4.hdf5", '325.2101135253906', '-221.2873992919922', '-19.490367889404297'],
    ["5.hdf5", '328.0667724609375', '-100.99529266357422', '-18.465534210205078']]

    assert avg_positions_data == expected_avg_positions, f"Unexpected average positions data: {avg_positions_data}"

    max_distances_data = read_csv(max_distances_csv)
    expected_max_distances = [
    ["File Name", "Device_792888129_0_Sensor_0"],
    ["1.hdf5", '280.263916015625'],
    ["2.hdf5", '315.810302734375'],
    ["3.hdf5", '347.91015625'],
    ["4.hdf5", '393.97943115234375'],
    ["5.hdf5", '343.83514404296875']]

    assert max_distances_data == expected_max_distances, f"Unexpected max distances data: {max_distances_data}"