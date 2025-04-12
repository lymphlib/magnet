"""Cell module.

Class hierarchy for mesh cells.
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.stats import uniform_direction
from magnet.geometric_utils import (shoelace_formula, polygon_centroid,
                                    tetrahedron_volume, tetrahedron_center,
                                    closest_to_segment, closest_to_face,
                                    convexHull_center, polygon_area_vector)

TOL = 1e-8


class Cell(ABC):
    """Single cell of a mesh.

    Parameters
    ----------
    nodes : list of int
        Global indices of the cell vertices.
    mesh_vertices : np.ndarray of float
        Coordiantes of all vertices of the mesh.
    faces : list of list of int
        Faces (or edges, in 2D) of the cell, each deascribed by the global Ids
        of its vertices.

    Attributes
    ----------
    Nodes : list of int
        Global Indices of the cell vertices.
    MeshVertices : np.ndarray of float
        Coordiantes of all vertices of the mesh.
    Faces : list of list of int
        Faces (or edges, in 2D) of the cell, each deascribed by the global Ids
        of its vertices.
    Vertices : np.ndarray of float
        Coordinates of the cell vertices (ordered according to local
        numbering).
    """

    def __init__(self, nodes=None, mesh_vertices=None, faces=None) -> None:
        self.Nodes = nodes
        # this will always be a shared reference to the same numpy array, no
        # data duplication is happening.
        self.MeshVertices = mesh_vertices
        if faces is not None:
            self.Faces = faces
        else:
            self._make_faces()

    @property
    def Vertices(self) -> np.ndarray:
        """Cell vertices coordinates."""
        return self.MeshVertices[self.Nodes]

    @abstractmethod
    def _make_faces(self) -> None:
        raise NotImplementedError("Must override method _make_faces")

    @abstractmethod
    def volume_center(self):
        raise NotImplementedError("Must override method volume_center")

    @abstractmethod
    def is_inside(self, p) -> bool:
        raise NotImplementedError("Must override method is_inside")

    @abstractmethod
    def inscribed_diameter(self, c=None) -> float:
        raise NotImplementedError("Must override method inscribed_diameter")


class Polygon(Cell):
    """Generic 2D polygonal cell."""
    def _make_faces(self) -> None:
        """Creates global indexing of face nodes based on local indexing."""
        self.Faces = [[self.Nodes[i-1], self.Nodes[i]]
                      for i in range(len(self.Nodes))]

    def volume_center(self):
        """Compute the area and centroid of the polygon.

        Parameters
        ----------
        None

        Returns
        -------
        volume : float
            The area of the polygon.
        centroid : np.ndarray of float
            Centroid of the polygon.
        """
        return self.area(), self.centroid()

    def area(self):
        """Compute the area of the polygon.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Area value.
        """
        return abs(shoelace_formula(self.Vertices))

    def centroid(self) -> np.ndarray:
        """Compute centroid coordinates of the polygon.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of shape (2,) corresponding to the centroid coordiantes.
        """
        return polygon_centroid(self.Vertices)

    def perimeter(self) -> float:
        """Compute the perimeter of the polygon.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Perimeter value.
        """
        perimeter = 0
        for i in range(len(self.Nodes)):
            perimeter += np.linalg.norm(self.Vertices[i]-self.Vertices[i-1])
        return perimeter

    def is_inside(self, p: np.ndarray) -> bool:
        """Check if a point is inside the polygon.

        Parameters
        ----------
        p : np.ndarray of float
            Point in question.

        Returns
        -------
        bool
            `True` if the point is inside the polygon, `False` otherwise.

        Notes
        -----
        Uses ray casting method; the ray is cast in a random direction.
        """
        intersections = 0
        ray = uniform_direction(2).rvs()

        for face in self.Faces:
            v = self.MeshVertices[face]
            # compute intersection between the ray and the edge line:
            A = np.column_stack((v[1]-v[0], -ray))
            x = np.linalg.solve(A, p-v[0])
            if x[1] > 0 and x[0] > 0 and x[0] < 1:
                intersections += 1

        # if the number of intersection is odd, the point is inside.
        return bool(intersections % 2)

    def inscribed_diameter(self, center: np.ndarray = None) -> float:
        """Compute the diameter of the inscribed circle of the cell.

        Computes (approximately) the diameter of the biggest circle
        contained in the polygon.

        Parameters
        ----------
        center : np.ndarray of flot, optional
            Center point of the circle for diameter computation. By default,
            uses the centroid of the polygon.

        Returns
        -------
        float
            The diameter of the inscribed cirlce.
        """
        center = self.centroid() if center is None else center
        if not self.is_inside(center):
            return 0
        else:
            return 2*min([np.linalg.norm(center-closest_to_segment(
                *self.MeshVertices[face], p=center))
                for face in self.Faces])

    def is_counterclockwise(self):
        """Check if polygon vertices are ordered counterclockwise.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            `True` if the vertices are ordered counterclockwise, `False` if
            clockwise.
        """
        signed_area = shoelace_formula(self.Vertices)
        return bool(signed_area > 0)

    def sort_nodes(self):
        """Sort nodes of a polygon on the perimeter.

        Reconstructs the order of nodes on the perimeter starting from the
        list of edges with arbitrary order. The resulting order can be either
        clockwise or counterclockwise. Updates `Nodes` attribute at the end.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # create dictionary with incident edges for each node
        incident_edges = {}
        for edge in self.Faces:
            a, b = edge
            if a not in incident_edges:
                incident_edges[a] = []
            incident_edges[a].append(b)

            if b not in incident_edges:
                incident_edges[b] = []
            incident_edges[b].append(a)

        # Traverse the perimeter of the polygon
        perimeter = []
        current_node = self.Faces[0][0]
        next_node = None
        visited_edges = set()

        while next_node != self.Faces[0][0]:  # different from starting node
            perimeter.append(current_node)

            # Find the next node by choosing the unvisited incident edge
            for neighbor in incident_edges[current_node]:
                if current_node < neighbor:
                    edge = (current_node, neighbor)
                else:
                    edge = (neighbor, current_node)
                if edge not in visited_edges:
                    next_node = neighbor
                    visited_edges.add(edge)
                    break
            current_node = next_node

        if len(perimeter) < len(self.Nodes):
            warnings.warn(
                'Polygon is not simply connected. Bad things will happen.')

        self.Nodes = perimeter


class Triangle(Polygon):
    """Triangular cell."""
    def _make_faces(self) -> None:
        """Creates global indexing of face nodes based on local indexing."""
        self.Faces = [[self.Nodes[0], self.Nodes[1]],
                      [self.Nodes[1], self.Nodes[2]],
                      [self.Nodes[2], self.Nodes[0]]]

    def area(self):
        """Compute the area of the triangle.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Area value.
        """
        return tetrahedron_volume(self.Vertices)

    def centroid(self):
        """Compute centroid coordinates of the triangle.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of shape (2,) corresponding to the centroid coordiantes.
        """
        return tetrahedron_center(self.Vertices)


class Quad(Polygon):
    pass


class Polyhedron(Cell):
    """Generic 3D polyhedral cell."""
    def _make_faces(self) -> None:
        raise NotImplementedError(
            "Not able to reconstruct faces of a generic polyhedron.")

    def volume_center(self):
        raise NotImplementedError(
            "Polyhedron volume and centroid not implemented.")

    def volume(self):
        """Compute the volume of the polyhedron.

        The volume is computed using the formula derived from the divergence
        theorem. Note that the face normals must be oriented consistently for
        this to work.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Volume value.
        """
        volume = 0
        for face in self.Faces:
            face_vertices = self.MeshVertices[face]
            area_vector = polygon_area_vector(face_vertices)
            volume += np.dot(area_vector, self.MeshVertices[0])
        return abs(volume)/3

    def is_inside(self, p):
        """Check if a point is inside the polyhedron.

        Parameters
        ----------
        p : np.ndarray of float
            Point in question.

        Returns
        -------
        bool
            True if the point is inside the polyhedron, False otherwise.

        Notes
        -----
        Uses ray casting method; the ray is cast in a random direction. Only
        supports triangular faces.
        """
        n_intersections = 0
        ray = uniform_direction(3).rvs()

        for face in self.Faces:
            v = self.MeshVertices[face]
            # compute intersection of ray with face plane
            n = np.cross(v[1]-v[0], v[2]-v[0])  # plane normal
            a = np.dot(n, v[0]-p)/np.dot(n, ray)
            if a > 0:
                intersection = p + a*ray  # intersection with face plane
                c = closest_to_face(v, intersection)
                # if the intersection with the plane is in the triangle:
                if all(np.abs(intersection-c) < TOL):
                    n_intersections += 1

        # if the number of intersection is odd, the point is inside.
        return bool(n_intersections % 2)

    def inscribed_diameter(self, center=None) -> float:
        """Compute the diameter of the inscribed sphere of the cell.

        Computes (approximately) the diameter of the biggest sphere
        contained in the polyhedron.

        Parameters
        ----------
        center : np.ndarray, optional
            Center point of the circle for diameter computation. By default,
            uses the centroid of the convex hull of the cell.

        Returns
        -------
        float
            The diameter of the inscribed sphere.

        Notes
        -----
        The inscribed diameter is computed by taking the centroid of the cell
        and then computing its minimum distance to each face of the polyhedron.
        If the face is a triangle, this is done geometrically by projection
        (cheaper), otherwise a constrained minimization problem is solved
        (slower and less accurate).
        """
        center = convexHull_center(self.Vertices) if center is None else center
        if not self.is_inside(center):
            return 0
        else:
            radius = np.inf
            for face in self.Faces:
                c = closest_to_face(self.MeshVertices[face], p=center)
                dist = np.linalg.norm(center - c)
                radius = min(radius, dist)
            return 2*radius

    def surface_area(self) -> float:
        """Compute the surface area of the cell.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Surface area value.
        """
        surface_area = 0
        for face in self.Faces:
            face_vertices = self.MeshVertices[face]
            face_area = np.linalg.norm(polygon_area_vector(face_vertices))
            surface_area += face_area
        return surface_area


class Tetrahedron(Polyhedron):
    def _make_faces(self) -> None:
        """Creates global indexing of face nodes based on local indexing."""
        self.Faces = [[self.Nodes[0], self.Nodes[1], self.Nodes[2]],
                      [self.Nodes[0], self.Nodes[1], self.Nodes[3]],
                      [self.Nodes[0], self.Nodes[2], self.Nodes[3]],
                      [self.Nodes[1], self.Nodes[2], self.Nodes[3]]]

    def volume_center(self):
        """Compute the volume and centroid of the tetrahedron.

        Parameters
        ----------
        None

        Returns
        -------
        volume : float
            The volume of the tetrahedron.
        centroid : np.ndarray of float
            Centroid of the tetrahedron.
        """
        return self.volume(), self.centroid()

    def volume(self):
        """Compute volume of the tetrahedron.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Volume value.
        """
        return tetrahedron_volume(self.Vertices)

    def centroid(self):
        """Compute centroid coordinates of the tetrahedron.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray of float
            Array of shape (3,) corresponding to the centroid coordiantes.
        """
        return tetrahedron_center(self.Vertices)


class Hexahedron(Polyhedron):
    """Creates global indexing of face nodes based on local indexing."""
    def _make_faces(self) -> None:
        self.Faces = [[self.Nodes[0], self.Nodes[1], self.Nodes[2], self.Nodes[3]],
                      [self.Nodes[0], self.Nodes[1], self.Nodes[5], self.Nodes[4]],
                      [self.Nodes[1], self.Nodes[2], self.Nodes[6], self.Nodes[5]],
                      [self.Nodes[2], self.Nodes[3], self.Nodes[7], self.Nodes[6]],
                      [self.Nodes[3], self.Nodes[0], self.Nodes[4], self.Nodes[7]],
                      [self.Nodes[4], self.Nodes[5], self.Nodes[6], self.Nodes[7]]]

    def volume_center(self) -> tuple[float, np.ndarray]:
        """Compute the volume and centroid of the hexahedron.

        Parameters
        ----------
        None

        Returns
        -------
        volume : float
            The volume of the hexahedron.
        centroid : np.ndarray of float
            Centroid of the hexahedron.

        Notes
        -----
        Centroid and volume are computed by dividing the hexahedron in 6
        tetrahedra using the 'long diagonal' (`v[1]`, `v[7]`).
        """
        # Division in 6 tetrahedra using the 'long diagonal' [1, 7]:
        v = self.Vertices
        tets = [v[[1, 0, 3, 7]],
                v[[1, 0, 4, 7]],
                v[[1, 5, 4, 7]],
                v[[1, 2, 3, 7]],
                v[[1, 2, 6, 7]],
                v[[1, 5, 6, 7]]]
        volume = 0
        centroid = np.zeros(3)
        for tet in tets:
            vol = tetrahedron_volume(tet)
            volume += vol
            centroid += vol*tetrahedron_center(tet)
        return volume, centroid/volume


class Pyramid(Polyhedron):
    def _make_faces(self) -> None:
        """Creates global indexing of face nodes based on local indexing."""
        self.Faces = [[self.Nodes[0], self.Nodes[1], self.Nodes[2], self.Nodes[3]],
                      [self.Nodes[0], self.Nodes[1], self.Nodes[4]],
                      [self.Nodes[1], self.Nodes[2], self.Nodes[4]],
                      [self.Nodes[2], self.Nodes[3], self.Nodes[4]],
                      [self.Nodes[3], self.Nodes[0], self.Nodes[4]]]

    def volume_center(self, v: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute volume and centroid of the pyramid with quadrilateral base.

        Parameters
        ----------
        None

        Returns
        -------
        volume : float
            The volume of the pyramid.
        centroid : np.ndarray of float
            Centroid of the pyramid.
        """
        v = self.Vertices
        tet1 = v[[0, 1, 3, 4]]
        tet2 = v[[1, 2, 3, 4]]
        vol1 = tetrahedron_volume(tet1)
        vol2 = tetrahedron_volume(tet2)
        volume = vol1 + vol2
        centroid = (vol1*tetrahedron_center(tet1)
                    + vol2*tetrahedron_center(tet2))/volume
        return volume, centroid
