import numpy as np
import matplotlib.pyplot as plt

segment_bit_length = 112
localization_bit_length = 12

def hash_with_hpp(vectors: np.ndarray, num_planes: int = segment_bit_length, seed: int = 42) -> np.ndarray:
    """
    Computes the hash for each vector using random hyperplanes.

    Parameters:
    vectors (np.ndarray): The input vectors to be hashed.
    num_planes (int): The number of random hyperplanes to generate. Defaults to segment_bit_length.

    Returns:
    np.ndarray: The computed hashes for each vector.
    """
    # Generate random hyperplanes
    np.random.seed(seed)
    hyperplanes = np.random.normal(size=(num_planes, vectors.shape[1]))
    # Compute the hash for each vector
    hashes = np.sign(np.dot(vectors, hyperplanes.T))
    # Replace -1 with 0 in hashes
    hashes = np.where(hashes == -1, 0, hashes)
    # Convert every element in hashes into an integer
    hashes = hashes.astype(int)
    # Convert to a hexadecimal string
    hashes = np.apply_along_axis(bit_array_to_hex, 1, hashes)

    return hashes

def latlon2xyz(lat: float, lon: float) -> np.ndarray:
    """
    Convert latitude and longitude coordinates to Cartesian coordinates (x, y, z).

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        np.ndarray: Array containing the Cartesian coordinates (x, y, z).
    """
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])

def convert_binary_to_hex(binary: str) -> str:
    """
    Converts a binary string to a hexadecimal string.

    Args:
        binary (str): The binary string to be converted.

    Returns:
        str: The hexadecimal representation of the binary string.
    """
    return hex(int(binary, 2))[2:]

def get_meridian_point_of_gc(normal_vector: np.ndarray) -> np.ndarray:
    """
    Calculates the intersection between the great circle defined by the given normal vector, and the prime meridian plane.
    
    Args:
        normal_vector (np.ndarray): The normal vector of the great circle.
        
    Returns:
        np.ndarray: The coordinates of the meridian point in xyz format.
    """
    
    theta = np.arctan2(normal_vector[0], normal_vector[1])
    # Replace invalid values with 0
    theta = np.where(np.isnan(theta), 0, theta)
    
    # Convert to xyz coordinates
    x = np.cos(theta)
    y = 0
    z = np.sin(theta)

    return np.array([x, y, z])

def compute_segment_position_hash(mid_point: np.ndarray, normal_vec: np.ndarray, random_angles: int = localization_bit_length, seed: int = 42) -> float:
    """
    Compute the segment position hash based on the given midpoint and normal vector.

    Parameters:
    mid_point (np.ndarray): The midpoint of the segment.
    normal_vec (np.ndarray): The normal vector of the segment.
    random_angles (int): The number of random angles to use for computing the hash. Default is 4.
    seed (int): The seed value for the random number generator. Default is 42.

    Returns:
    float: The computed segment position hash.

    """
    S = np.apply_along_axis(get_meridian_point_of_gc, 1, normal_vec)
    # S = get_meridian_point_of_gc(normal_vec)
    # Compute the angle between the midpoint and the prime meridian point
    dot_product = np.sum(mid_point.T * S, axis=1)
    cross_product = np.cross(mid_point.T, S)
    sign_of_dot_product = np.sign(cross_product[:, 2])
    angle = sign_of_dot_product * np.arccos(dot_product)
    # for angle < 0, add 2pi
    angle = angle + 2*np.pi * (angle < 0)
    np.random.seed(seed)
    key_angles = np.random.uniform(0, 2*np.pi, random_angles) # 16 random angles
    # Compute the hash: for each key_angle, check if the angle is within +-pi/2 radians of the key_angle
    hash = np.zeros((mid_point.shape[1], random_angles)) # mid_point: 3x10000
    for i in range(random_angles):
        hash[:, i] = (np.abs(angle - key_angles[i]) % (np.pi)) <= np.pi/2
    hash = np.dot(hash, 2**np.arange(random_angles))
    hash = [hex(int(h))[2:] for h in hash]
    return hash

def bit_array_to_hex(bit_array: np.ndarray) -> str:
    """
    Converts a bit array to a hexadecimal string.

    Args:
        bit_array (np.ndarray): The bit array to be converted.

    Returns:
        str: The hexadecimal representation of the bit array.
    """
    binary_str = ''.join(bit_array.astype(str))
    padded_binary_str = binary_str.zfill(len(binary_str) + (4 - len(binary_str) % 4) % 4)
    chunks = [padded_binary_str[i:i+4] for i in range(0, len(padded_binary_str), 4)]
    hex_str = ''.join([hex(int(chunk, 2))[2:] for chunk in chunks])
    return hex_str
