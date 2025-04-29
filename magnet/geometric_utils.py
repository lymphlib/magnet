"""Geometric utilities module.

Contains functions used for geometric operations, e.g. volume, centroid and
distance computations.
"""

import numpy as np
from math import factorial
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def tetrahedron_center(vertices: np.ndarray) -> np.ndarray:
    """Compute the baricenter of a simplex.

    Takes the mean of the coordinates of the vertices.

    Parameters
    ----------
    vertices : np.ndarray of float
        Array of shape (dim + 1, dim), where dim is the dimension of the
        considered space, containing the coordinates of the vertices of the
        simplex.

    Returns
    -------
    np.ndarray of float
        The coordinates of the baricenter.
    """
    return np.mean(vertices, axis=0)


def tetrahedron_volume(vertices: np.ndarray) -> float:
    """Compute the volume of a simplex.

    The volume is computed using the Cayley-Menger determinant.

    Parameters
    ----------
    vertices : np.ndarray v
        Array of shape (dim + 1, dim), where dim is the dimension of the
        considered space, containing the coordinates of the vertices of the
        simplex.

    Returns
    -------
    float
        The volume of the simplex.
    """
    return np.abs(
        np.linalg.det([np.append(point, [1]) for point in vertices])
    ) / factorial(
        vertices.shape[-1]
    )  # /factorial(dimension)


def shoelace_formula(v: np.ndarray) -> float:
    """Compute the area of a 2D polygon.

    Computes the signed area of a polygon; the area is positive if the
    vertices are ordered counter-clockwise, negative if clockwise instead.

    Parameters
    ----------
    v : np.ndarray of float
        array of shape (N, 2), where N is the number of vertices of the
        polygon, containing the coordinates of its vertices ordered as they
        are encountered on the perimeter.

    Returns
    -------
    float
        The computed area.
    """
    n_vert = len(v)
    return 0.5 * np.sum(
        np.array([v[i - 1, 0] * v[i, 1] - v[i, 0] * v[i - 1, 1] for i in range(n_vert)])
    )


def polygon_centroid(v: np.ndarray) -> np.ndarray:
    """Compute the centroid of a 2D polygon.

    Parameters
    ----------
    v : np.ndarray of float
        array of shape (N, 2), where N is the number of vertices of the
        polygon, containing the coordinates of its vertices ordered as they
        are encountered on the perimeter (clockwise or not is indifferent).

    Returns
    -------
    float
        The computed area.
    """
    n_vert = len(v)
    cross_products = np.array(
        [v[i - 1, 0] * v[i, 1] - v[i, 0] * v[i - 1, 1] for i in range(n_vert)]
    )
    area = 0.5 * np.sum(cross_products)  # shoelace formula
    return np.array(
        [
            np.sum(
                np.array(
                    [(v[i - 1, 0] + v[i, 0]) * cross_products[i] for i in range(n_vert)]
                )
            ),
            np.sum(
                np.array(
                    [(v[i - 1, 1] + v[i, 1]) * cross_products[i] for i in range(n_vert)]
                )
            ),
        ]
    ) / (6 * area)


def polygon_area_vector(v: np.ndarray) -> np.ndarray:
    """Compute area vector of a polygon in 3D space.

    Compute the vector normal to the polygon's plane having magnitude equal to
    its area; the direction depends on the arrangement of the nodes (clockwise
    or counterclockwise).

    Parameters
    ----------
    v : np.ndarray
        Array of shape (N, 3), where N is the number of vertices of the
        polygon, containing the polygon vertices oordinates.

    Returns
    -------
    np.ndarray
        Polygon area vector.
    """
    area = sum([np.cross(v[i - 1], v[i]) for i in range(len(v))])
    return area / 2


def convexHull_center(vertices: np.ndarray) -> np.ndarray:
    """Compute the centroid of the convex hull of a set of points.

    Parameters
    ----------
    vertices : np.ndarray of float
        array of shape (N, 2), where N is the number of vertices of the
        polygon, containing the coordinates of its vertices.

    Returns
    -------
    float
        The computed area.

    Notes
    -----
    In the 2D case, `polygon_centroid` is used for the computation. In the
    general case, we first find the Delaunay triangulation and then take a
    weighted average of all the simplices to find the centroid.
    """
    dim = vertices.shape[-1]
    vertices = vertices[ConvexHull(vertices).vertices]
    # if the dimension is 2, vertices are automatically sorted counter
    # clockwise by ConvexHull and we can use the shoelace formula.
    if dim == 2:
        return polygon_centroid(vertices)
    else:
        T = Delaunay(vertices)
        total_volume = 0
        centre = np.zeros(dim)
        for simplex in T.simplices:
            vol = tetrahedron_volume(vertices[simplex])
            centre += vol * tetrahedron_center(vertices[simplex])
            total_volume += vol
        return centre / total_volume


def maximum_sq_distance(points: np.ndarray) -> float:
    """Compute the squared diameter of a set of points.

    Computes the maximum squared distance between any two points in the set
    `points`.

    Parameters
    ----------
    points : np.ndarray of float
        Coordinates of the points (each row corresponds to a point).

    Returns
    -------
    float
        maximum squared distance

    Notes
    -----
    The two points at maximum distance are always vertices of the convex hull
    of the set of points. If the number of points is large, the convex hull is
    computed first to speed up computation.
    """
    if len(points) < 750:
        return pdist(points, metric="sqeuclidean").max()
    else:
        return pdist(points[ConvexHull(points).vertices], metric="sqeuclidean").max()


def project(x1: np.ndarray, x2: np.ndarray, p: np.ndarray) -> float:
    """Project a point onto a line.

    Computes the projection of point `p` onto the line passing through the
    points `x1`, `x2`.

    Parameters
    ----------
    x1, x2 : np.ndarray of float
        Coordinates of the two points defining the line.
    p : np.ndarray of float
        Coordinates of the point to project.

    Returns
    -------
    float
        Projection value.
    """
    return np.dot(p - x1, x2 - x1) / np.dot(x2 - x1, x2 - x1)


def _point_at(A, B, t):
    """Point on line (`A`, `B`) individuated by scalar `t`."""
    return A + t * (B - A)


def closest_to_segment(x1: np.ndarray, x2: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute the point of a segment closest to 1a given point.

    Computes the the point of the segment of extremes `x1`, `x2` that is
    closest to the point `p`.

    Parameters
    ----------
    x1, x2 : np.ndarray of float
        Coordinates of the extremes of the segment.
    p : np.ndarray of float
        Coordinates of the point of interest.

    Returns
    -------
    np.ndarray of float
        The coordinates of the point closest to `p`.
    """
    alpha = project(x1, x2, p)
    # If alpha is outside (0,1), then the projection lies outside of the
    # segment and the closest point is one of the extremes.
    alpha = max(0, min(1, alpha))
    return _point_at(x1, x2, alpha)


def closest_to_triangle(triangle_v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute the point of a triangle closest to a given point.

    Computes the the point of the triangle of vertices `triangle_v` that is
    closest to the point `p`.

    Parameters
    ----------
    triangle_v : np.ndarray of float
        Array of shape (3, dim) of the coordinates of the vertices of the
        triangle.
    p : np.ndarray of float
        Coordinates of the point of interest.

    Returns
    -------
    np.ndarray of float
        The coordinates of the point closest to `p`.
    """
    A = triangle_v[0]
    B = triangle_v[1]
    C = triangle_v[2]

    # Case 1: one of the vertices is closest:
    uab = project(A, B, p)
    uca = project(C, A, p)
    if uca > 1 and uab < 0:
        return A
    ubc = project(B, C, p)
    if uab > 1 and ubc < 0:
        return B
    if ubc > 1 and uca < 0:
        return C

    # Case 2: the closest point is on one of the edges:
    tri_normal = np.cross(B - A, C - A)
    if 0 < uab and uab < 1 and np.dot(np.cross(tri_normal, B - A), p - A) < 0:
        return _point_at(A, B, uab)

    if 0 < ubc and ubc < 1 and np.dot(np.cross(tri_normal, C - B), p - B) < 0:
        return _point_at(B, C, ubc)

    if 0 < uca and uca < 1 and np.dot(np.cross(tri_normal, A - C), p - C) < 0:
        return _point_at(C, A, uca)

    # If we are not in any of the above cases, the closest point is in the
    # interior and we simply project onto the triangle plane:
    return p - np.dot(p - A, tri_normal) / np.dot(tri_normal, tri_normal) * tri_normal


def closest_to_face(face_verts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute the point of a face closest to a given point.

    Utility function that selects `closest_to_triangle` if the face has 3
    sides, or  `linear_constrained_least_squares` if it has more than 3, to
    compute the point on the face closest to `p`.

    Parameters
    ----------
    face_verts : np.ndarray of float
        Array of shape (N, 3) of the coordinates of the vertices of the
        face.
    p : np.ndarray of float
        Coordinates of the point of interest.

    Returns
    -------
    np.ndarray of float
        The coordinates of the point closest to `p`.
    """
    n_vert = len(face_verts)
    if n_vert == 3:
        return closest_to_triangle(face_verts, p)
    else:
        face_verts = face_verts.T
        return face_verts @ linear_constrained_least_squares(
            face_verts,
            p,
            np.ones(n_vert),
            1,
            np.zeros(n_vert),
            np.ones(n_vert),
            np.mean(face_verts, axis=1),
        )


def linear_constrained_least_squares(
    C: np.ndarray,
    d: np.ndarray,
    Aeq: np.ndarray,
    beq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
) -> np.ndarray:
    """Solve linear constrained least squares problem.

    The problem solved is : minimize 0.5 * ||`C`@x - `d`||**2 subject to the
    equality constraints `Aeq`@x = `beq`, and to the bounds on variable x
    `lb` <= x <= `ub` (inequalities are intended component-wise).

    Parameters
    ----------
    C : np.ndarray of float
        Design matrix.
    d : np.ndarray of float
        Target vector.
    Aeq: np.ndarray of float
        Equality constraint matrix.
    beq: np.ndarray of float
        Equality constraint right hand side vector.
    lb, ub : np.ndarray of float
        Lower and upper bounds on parameters.
    x0: np.ndarray of float
        Initial guess.

    Returns
    -------
    np.ndarray of float
        Result of the minimization.

    Notes
    -----
    The solver used in this function is `scipy.optimize.minimize', which is a
    very general optimizer, so it is not optimized for this problem.
    As such, the function can be slow.
    """

    def objective(x):
        return 0.5 * np.linalg.norm(np.dot(C, x) - d) ** 2

    constraints = [{"type": "eq", "fun": lambda x: beq - np.dot(Aeq, x)}]
    bounds = [(lb[i], ub[i]) for i in range(len(lb))]
    result = minimize(objective, x0, constraints=constraints, bounds=bounds)

    return result.x
