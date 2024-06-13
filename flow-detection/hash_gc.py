import numpy as np
def generate_hash_functions(num_planes, dim, seed=None):
    """
    Generate random hyperplanes for hashing.
    
    Parameters:
    num_planes (int): Number of random hyperplanes to generate.
    dim (int): Dimension of the input vectors.
    seed (int): Seed for random number generator to ensure reproducibility.
    
    Returns:
    array-like: Random hyperplanes of shape (num_planes, dim).
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.randn(num_planes, dim)

def hyperplane_lsh(u, v, num_planes=10, seed=None):
    """
    Perform hyperplane LSH between two vectors u and v.
    
    Parameters:
    u (array-like): First input vector.
    v (array-like): Second input vector.
    num_planes (int): Number of random hyperplanes to use for hashing.
    seed (int): Seed for random number generator to ensure reproducibility.
    
    Returns:
    bool: True if u and v hash to the same bucket, False otherwise.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create random hyperplanes
    dim = len(u)
    hyperplanes = np.random.randn(num_planes, dim)
    
    # Compute the hash codes for both vectors
    hash_u = np.sign(np.dot(hyperplanes, u))
    hash_v = np.sign(np.dot(hyperplanes, v))
    
    # Compare the hash codes
    return np.array_equal(hash_u, hash_v)

# Example usage
u = np.array([1.0, 2.0, 3.0])
v = np.array([1.1, 2.1, 3.1])

same_bucket = hyperplane_lsh(u, v, num_planes=10, seed=42)
print(f"Vectors u and v hash to the same bucket: {same_bucket}")
