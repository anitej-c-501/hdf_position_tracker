import os
import csv
import h5py
import numpy as np
from file_handler import validate_folder, list_hdf5_files, open_hdf5_file
from utils import compute_average_position, compute_max_distance, format_csv_data

def process_hdf5_file(file_path):
    """
    Process an individual HDF5 file to compute metrics for each sensor.
    :param file_path: Path to the HDF5 file
    :return: Tuple (average_positions, max_distances, sensor_identifiers)
    """
    average_positions = []
    max_distances = []
    sensor_identifiers = []

    print("Checkpoint 1")

    with open_hdf5_file(file_path) as hdf_file:
        for device in hdf_file.keys():
            device_group = hdf_file[device]
            print(device_group)
            print("Checkpoint 2")
            # Ensure "Position" dataset exists
            if "Position" not in device_group:
                print(f"Warning: Skipping device '{device}' in file '{file_path}' (no 'Position' data).")
                continue
            print("Checkpoint 3")
            position_data = device_group["Position"][:]

            # Ensure data has the expected shape
            if not isinstance(position_data, np.ndarray):
                raise ValueError(f"'Position' data in device '{device}' is not a NumPy array.")

            if len(position_data.shape) != 3 or position_data.shape[2] != 3:
                raise ValueError(f"Unexpected 'Position' shape {position_data.shape} in device '{device}'.")

            for sensor_index in range(position_data.shape[1]):
                sensor_data = position_data[:, sensor_index, :]

                # Ensure sensor_data is a valid array
                if sensor_data.size == 0:
                    print(f"Warning: Sensor {sensor_index} in device '{device}' has no data.")
                    continue

                try:
                    avg_position = compute_average_position(sensor_data)
                    max_distance = compute_max_distance(sensor_data)

                    average_positions.extend(avg_position)
                    max_distances.append(max_distance)

                    sensor_identifiers.append(f"{device}_Sensor_{sensor_index}")
                except Exception as e:
                    print(f"Error processing sensor {sensor_index} in device '{device}': {e}")
                    continue

    return average_positions, max_distances, sensor_identifiers



def write_csv(output_file, headers, rows):
    """
    Write data to a CSV file.
    :param output_file: Path to the output CSV file
    :param headers: List of column headers
    :param rows: List of rows to write
    """
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)


def process_data(input_folder, output_folder):
    """
    Process all HDF5 files in the input folder and generate CSV reports.
    :param input_folder: Path to the folder containing input HDF5 files
    :param output_folder: Path to the folder to save output CSV files
    """
    validate_folder(input_folder)
    validate_folder(output_folder, create_if_missing=True)

    hdf5_files = list_hdf5_files(input_folder)

    avg_position_rows = []
    max_distance_rows = []
    all_sensor_identifiers = set()

    # First pass: Collect all unique sensor identifiers
    for file_name in hdf5_files:
        file_path = os.path.join(input_folder, file_name)
        try:
            _, _, sensors = process_hdf5_file(file_path)
            all_sensor_identifiers.update(sensors)
        except Exception as e:
            print(f"Warning: Skipping file '{file_name}' due to error: {e}")

    print("Finished first pass")
    all_sensor_identifiers = sorted(all_sensor_identifiers)

    # Second pass: Process files and generate rows
    for file_name in hdf5_files:
        file_path = os.path.join(input_folder, file_name)
        try:
            avg_positions, max_distances, sensors = process_hdf5_file(file_path)

            # Convert `sensors` to a regular Python list if it's a NumPy array
            if isinstance(sensors, np.ndarray):
                sensors = sensors.tolist()

            # Map results to global identifiers
            avg_position_row = [file_name]
            max_distance_row = [file_name]

            for sensor in all_sensor_identifiers:
                if sensor in sensors:
                    sensor_index = sensors.index(sensor)
                    try:
                        # Add average position values
                        avg_position_row.extend(
                            [float(avg_positions[sensor_index * 3 + i]) for i in range(3)]
                        )
                        # Add max distance value
                        max_distance_row.append(float(max_distances[sensor_index]))
                    except (IndexError, ValueError) as e:
                        print(f"Error processing sensor '{sensor}' in file '{file_name}': {e}")
                        avg_position_row.extend([float('nan'), float('nan'), float('nan')])
                        max_distance_row.append(float('nan'))

                else:
                    # Sensor not found; fill with "NA"
                    avg_position_row.extend([float('nan'), float('nan'), float('nan')])
                    max_distance_row.append(float('nan'))

            avg_position_rows.append(avg_position_row)
            max_distance_rows.append(max_distance_row)

        except Exception as e:
            print(f"Warning: Skipping file '{file_name}' due to error: {e}")

    # Generate headers
    avg_position_headers = ["File Name"] + [
        f"{sensor}_{axis}" for sensor in all_sensor_identifiers for axis in ["X", "Y", "Z"]
    ]
    max_distance_headers = ["File Name"] + all_sensor_identifiers

    # Write CSVs
    write_csv(os.path.join(output_folder, "average_positions.csv"), avg_position_headers, avg_position_rows)
    write_csv(os.path.join(output_folder, "max_distances.csv"), max_distance_headers, max_distance_rows)

