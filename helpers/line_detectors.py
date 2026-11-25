"""
Functions that find a straight line in a set of edge points.

A line is stored in normal form (a, b, c):  a*x + b*y + c = 0,  with a^2 + b^2 = 1.
Because (a, b) has length 1, |a*x + b*y + c| is the distance from (x, y) to the line in pixels.
"""

import numpy as np


rng = np.random.default_rng()


def edge_points(edges):
    """All edge pixels of a Canny image as an (N, 2) array of (x, y)."""
    ys, xs = np.nonzero(edges)              # NumPy gives (row, col) = (y, x)
    return np.column_stack((xs, ys))        # swap once, use (x, y) from here on


def line_through(p1, p2):
    """Line through two points, in normal form (a, b, c)."""
    dx, dy = p2 - p1
    length = np.hypot(dx, dy)
    a, b = -dy / length, dx / length        # direction rotated 90 degrees = unit normal
    c = -(a * p1[0] + b * p1[1])            # p1 must lie on the line
    return a, b, c


def distances(points, line):
    """Distance in pixels from every point to the line."""
    a, b, c = line
    return np.abs(a * points[:, 0] + b * points[:, 1] + c)


def ransac_line(points, iterations, threshold, step=1):
    """Line with the most inliers. Support is counted on every `step`-th point only.
    Returns (line, inlier mask over all points), or (None, None) if there are too few points."""
    support_points = points[::step]
    if len(support_points) < 2:
        return None, None

    best_line = None
    best_count = 0
    for _ in range(iterations):
        i, j = rng.choice(len(support_points), size=2, replace=False)
        line = line_through(support_points[i], support_points[j])
        count = np.count_nonzero(distances(support_points, line) < threshold)
        if count > best_count:
            best_line = line
            best_count = count

    if best_line is None:
        return None, None

    inliers = distances(points, best_line) < threshold
    return best_line, inliers


def least_squares_line(points):
    """Line that minimises the perpendicular distances to the points (total least squares)."""
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean)
    a, b = vt[1]                            # direction of least spread = the normal
    c = -(a * mean[0] + b * mean[1])
    return a, b, c


def angle_between(line1, line2):
    """Angle in degrees between two lines (0 = parallel). The sign of (a, b) does not matter."""
    cos = abs(line1[0] * line2[0] + line1[1] * line2[1])    # dot product of the two unit normals
    return np.degrees(np.arccos(min(cos, 1.0)))


def iterations_needed(p, outlier_ratio):
    """Number of RANSAC iterations N to draw one clean pair with probability p."""
    clean_pair = (1 - outlier_ratio) ** 2
    if clean_pair >= 1:
        return 1
    if clean_pair <= 0:
        return float("inf")
    return np.log(1 - p) / np.log(1 - clean_pair)
