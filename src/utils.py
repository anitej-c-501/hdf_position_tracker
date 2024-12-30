import numpy as np

def compute_average_position(data):
    """
    Compute the average XYZ position for a dataset.
    :param data: Numpy array of shape [samples, 3]
    :return: List [mean_x, mean_y, mean_z]
    """
    return data.mean(axis=0).tolist()

def compute_max_distance(data):
    """
    Compute the maximum Euclidean distance for a dataset.
    :param data: Numpy array of shape [samples, 3]
    :return: Maximum Euclidean distance as a float
    """
    return np.linalg.norm(data, axis=1).max()

def format_csv_data(file_name, results):
    """
    Format results for CSV output.
    :param file_name: Name of the file being processed
    :param results: List of computed results for the file
    :return: List representing a row in the CSV
    """
    return [file_name] + results
