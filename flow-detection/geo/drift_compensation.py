import numpy as np
from scipy.optimize import fsolve

# Earth's radius
Re = 6371e3 # m

def latlon2xyz(lat, lon):
    """Converts latitude and longitude to cartesian coordinates

    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees

    Returns:
        float: x, y, z coordinates
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = Re * np.cos(lat) * np.cos(lon)
    y = Re * np.cos(lat) * np.sin(lon)
    z = Re * np.sin(lat)
    return np.array([x, y, z])

def get_true_north_pointing_vector(P: np.ndarray) -> np.ndarray:
    """Get the true north pointing vector at a given point P

    Args:
        P (np.ndarray): Point P in cartesian coordinates

    Returns:
        np.ndarray: True north pointing vector
    """
    x, y, z = P
    north = np.array([-x * z, -y * z, x**2 + y**2])
    return north / np.linalg.norm(north)

def course_equation(vars, xp, yp, zp, xn, yn, zn, crs):
    xc, yc, zc = vars
    eq1 = xp * xc + yp * yc + zp * zc # Course vector is perpendicular to the OP vector
    eq2 = xn * xc + yn * yc + zn * zc - np.cos(np.deg2rad(crs)) # Course vector forms an angle of crs with the true north vector
    eq3 = xc**2 + yc**2 + zc**2 - 1 # Course vector is a unit vector
    return eq1, eq2, eq3

def get_course_vector(P: np.ndarray, crs: float) -> np.ndarray:
    """Get the course vector at a given point P

    Args:
        P (np.ndarray): Point P in cartesian coordinates
        crs (float): Course in degrees

    Returns:
        np.ndarray: Course vector
    """
    north = get_true_north_pointing_vector(P)
    xp, yp, zp = P / np.linalg.norm(P)
    xn, yn, zn = north
    
    if crs > 0 and crs < 180:
        c0 = np.cross(north, P) / np.linalg.norm(np.cross(north, P)) # initial guess is the East vector
    else:
        c0 = np.cross(P, north) / np.linalg.norm(np.cross(P, north)) # initial guess is the West vector
    
    c = fsolve(course_equation, c0, args=(xp, yp, zp, xn, yn, zn, crs))

    
    return c / np.linalg.norm(c)

def get_second_vector_in_GC(P: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Get the second vector in the great circle defined by the course vector

    Args:
        P (np.ndarray): Point P in cartesian coordinates
        c (np.ndarray): Course vector

    Returns:
        np.ndarray: Second vector in the great circle
    """
    s = np.cross(np.cross(P, c), P)
    return s / np.linalg.norm(s)

def get_course_vector_on_GC(P: np.ndarray, s: np.ndarray, P0: np.ndarray) -> np.ndarray:
    """Get the course vector on the great circle defined by the second vector

    Args:
        P (np.ndarray): Point P in cartesian coordinates (perturbed P0)
        s (np.ndarray): Second vector in the great circle
        P0 (np.ndarray): The original point P in cartesian coordinates, to ensure the course vector is pointing in the right direction

    Returns:
        np.ndarray: Course vector on the great circle
    """
    normal_vec_of_GC = np.cross(P0, s)
    c = np.cross(normal_vec_of_GC, P)
    return c/np.linalg.norm(c)

def get_track_drift_rate(lat: float, lon: float, initial_crs: float):
    """Get the track drift rate at a given point

    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees
        initial_crs (float): Initial course in degrees

    Returns:
        drift_rate: Track drift rate in degrees per kilometer travel along the great-circle
        
    Example:
        drift_rate = get_track_drift_rate(37.7749, -122.4194, 45)
    """
    P = latlon2xyz(lat, lon) # Point P in cartesian coordinates
    c = get_course_vector(P, initial_crs) # the course vector at P
    s = get_second_vector_in_GC(P, c) # the second vector in the great circle defined by the course vector
    # s will be in the same direction as c
    
    km_for_diff = 10
    theta = km_for_diff * 1e3/Re # 100 km for differentiation, central angle
    P_prime = P * np.cos(theta) + s * Re * np.sin(theta)

    n_prime = get_true_north_pointing_vector(P_prime)
    c_prime = get_course_vector_on_GC(P_prime, s, P)
    angle = np.rad2deg(np.arccos(np.dot(n_prime, c_prime)))
    # We must differentiate between the two possible angles: negative and positive since the dot product is symmetric (does not reveal the direction of multiplication)
    # The idea is to compute the cross product between n_prime and c_prime, if the product is in the same direction as P', then the angle is positive (<180)
    # Otherwise, the angle is negative (>180)
    crx_nc = np.cross(n_prime, c_prime)
    if np.dot(crx_nc, P_prime) > 0:
        angle = 360 - angle

    delta_angle = angle - initial_crs
    # handle discontinuity at 0/360 degrees and preventing 180 degrees turn
    if delta_angle > 180: # like 10 -> 350
        delta_angle = delta_angle - 360
    elif delta_angle < -180: # like 350 -> 10
        delta_angle = delta_angle + 360

    d_angle = delta_angle / km_for_diff # degrees per kilometer travel
    return d_angle
