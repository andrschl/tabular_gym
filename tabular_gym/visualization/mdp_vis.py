"""
This script contains functions for the visualization of a 2-state-2-action MDP.
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import kl_div
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from numpy.linalg import lstsq, norm
from ..utils.geometric_tools import *
from einops import rearrange, repeat, einsum

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Iterable



########################################################################################################################
# geometric tools

class Plane:
    """
    Class for a hyperplane.
    """
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal

class Line:
    """
    Class for a hyperplane.
    """
    def __init__(self, point, direction):
        self.point = point
        self.direction = direction


ONB = 1/math.sqrt(2) * np.array([[1/math.sqrt(2), 1/math.sqrt(2), 1/math.sqrt(2), 1/math.sqrt(2)],
                                [1, 0, -1, 0],
                                [0, 1, 0, -1],
                                [1/math.sqrt(2), -1/math.sqrt(2), 1/math.sqrt(2), -1/math.sqrt(2)]]).T

def project_and_rotate(x, angle=0.0):
    # P = np.array([[1, 0, 0, 0],
    #           [0, 1, 0, 0],
    #           [0, 0, 0, 1],
    #           [0, 0, 1, 0]])
    # V = ONB @ P
    V = ONB
    # R = Rotation.from_rotvec([0, 0 , 0]).as_matrix()
    R = Rotation.from_rotvec([0, angle, 0]).as_matrix()
    x = V.T @ x
    return R @ x[1:]


def order_vertices(vertices):
    """
    Orders vertices of tetragon.
    :param vertices: ndarray, shape (4, 2).
    :return:
    """

    np.zeros_like(vertices)
    v01 = vertices[1] - vertices[0]
    v02 = vertices[2] - vertices[0]
    v03 = vertices[3] - vertices[0]
    a12 = math.acos(np.dot(v01, v02)/(norm(v01) * norm(v02)))
    a13 = math.acos(np.dot(v01, v03)/(norm(v01) * norm(v03)))
    a23 = math.acos(np.dot(v02, v03)/(norm(v02) * norm(v03)))
    if a12 > max(a23, a13):
        new_vertices = vertices.copy()
        new_vertices[2] = vertices[3]
        new_vertices[3] = vertices[2]
        return new_vertices

    if a23 > max(a12, a13):
        new_vertices = vertices.copy()
        new_vertices[1] = vertices[2]
        new_vertices[2] = vertices[1]
        return new_vertices
    return vertices


def plane_intersection(plane1, plane2):

    point1, normal1 = plane1.point, plane1.normal
    point2, normal2 = plane2.point, plane2.normal
    b1, b2 = np.dot(normal1, point1), np.dot(normal2, point2)
    b = np.array([b1, b2])
    A = np.vstack([normal1, normal2])

    # Check if the planes are parallel
    if np.linalg.matrix_rank(A, tol=1e-5) < 2:
        if np.allclose(b1, b2):
            return "Planes are coincident"
        else:
            return "No intersection (planes are parallel)"

    # Find the direction vector of the line of intersection
    direction = np.cross(normal1, normal2)

    # Find a point on the line of intersection
    # Solving for one variable (e.g., x=0) and calculating y and z
    if np.linalg.cond(A[:,1:]) < 1 / sys.float_info.epsilon:
        x = 0
        sol = np.linalg.solve(A[:, 1:], b).flatten()
        point = np.concatenate([[x], sol])
    elif np.linalg.cond(A[:, [0, 2]]) < 1 / sys.float_info.epsilon:
        y = 0
        sol = np.linalg.solve(A[:, [0, 2]], b).flatten()
        point = np.array([sol[0], y, sol[1]])
    else:
        z = 0
        sol = np.linalg.solve(A[:,:2], b).flatten()
        point = np.concatenate([sol, [z]])

    return Line(point, direction)

def line_triangle_intersection(line, v0, v1, v2):

    # unpack line
    p0 = line.point
    d = line.direction

    # Calculate plane's normal vector
    n = np.cross(v1 - v0, v2 - v0)

    # Check if the line and plane are parallel
    dot = np.dot(n, d)
    if np.abs(dot) < 1e-6:
        return None

    # Calculate the distance along the line to the intersection point
    t = np.dot(n, v0 - p0) / dot

    # Get the point of intersection
    p = p0 + t * d

    # Check if the intersection point is inside the triangle
    if (np.dot(np.cross(v1 - v0, p - v0), n) >= 0 and
        np.dot(np.cross(v2 - v1, p - v1), n) >= 0 and
        np.dot(np.cross(v0 - v2, p - v2), n) >= 0):
        return p
    return None


def line_tetrahedron_intersection(line, tetrahedron_vertices):
    # Unpack the vertices
    v0, v1, v2, v3 = tetrahedron_vertices

    # Check intersection for each face of the tetrahedron
    intersections = []
    for face in [(v0, v1, v2), (v0, v1, v3), (v0, v2, v3), (v1, v2, v3)]:
        intersection = line_triangle_intersection(line, *face)
        if intersection is not None:
            intersections.append(intersection)

    # Return the unique intersection points
    return np.unique(intersections, axis=0)


def plane_tetrahedron_intersection(plane, tetrahedron_vertices):
    # This code is generated by chatGPT

    intersection_points = []

    # Generate all possible combinations of edge indices
    edge_indices = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    for v1_index, v2_index in edge_indices:
        v1 = tetrahedron_vertices[v1_index]
        v2 = tetrahedron_vertices[v2_index]

        d1 = np.dot(plane.normal, v1) - np.dot(plane.normal, plane.point)
        d2 = np.dot(plane.normal, v2) - np.dot(plane.normal, plane.point)

        if d1 * d2 <= 0:
            t = np.abs(d1) / (np.abs(d1) + np.abs(d2))
            intersection_point = v1 + t * (v2 - v1)
            intersection_points.append(intersection_point)


    intersection_points = np.array(intersection_points)
    # print('intersection points', intersection_points)
    if len(intersection_points) == 4:
        intersection_points = order_vertices(intersection_points)
        # print(intersection_points)

    return intersection_points

########################################################################################################################
# plotting tools

def plot_occ_space(env, A=None, mu0=None, show=False, ax=None, fig=None, angle=0.0, color=None, name=None):

    # Define the plane normal, plane point, and tetrahedron vertices
    A = env.A_matrix() if A is None else A
    mu0 = lstsq(A.T, (1 - env.gamma) * env.nu0, rcond=None)[0] if mu0 is None else mu0
    color = 'darkgray' if color is None else color
    # print('plane normal: ', project_and_rotate(A, angle=angle).T)
    plane = Plane(project_and_rotate(mu0, angle=angle), project_and_rotate(A, angle=angle).T[0])
    tetrahedron_vertices = project_and_rotate(np.eye(4), angle=angle).T

    # Calculate the intersection points
    intersection_points = plane_tetrahedron_intersection(plane, tetrahedron_vertices)

    # Plot the tetrahedron
    hull = ConvexHull(tetrahedron_vertices)
    points = hull.points


    if fig is None:

        # Create the tetrahedron plot
        tetrahedron = go.Mesh3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            color='lightgray',
            opacity=0.1
        )

        # Create figure
        fig = go.Figure(data=[tetrahedron])

        # Set axis labels
        fig.update_layout(
            scene=dict(
                xaxis_title='v_1',
                yaxis_title='v_2',
                zaxis_title='v_3'
            ),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )

    # Plot intersection points
    scatter = go.Scatter3d(
        x=intersection_points[:, 0],
        y=intersection_points[:, 1],
        z=intersection_points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue'
        ),
        showlegend=False
    )
    fig.add_trace(scatter)

    # Occupancy measure set
    poly3d = go.Mesh3d(
        x=intersection_points[:, 0],
        y=intersection_points[:, 1],
        z=intersection_points[:, 2],
        alphahull=0,
        color=color,
        opacity=0.3,
        flatshading=True,
        name=name,
        showlegend=True
    )
    fig.add_trace(poly3d)

    # Show the plot
    if show:
        fig.show()
    return fig, plane


def plot_intersecting_lines(planes, fig, angle=0.0, color='black', linewidth=1):

    # plot the intersecting lines between the planes
    lines = []
    for i, plane1 in enumerate(planes):
        for plane2 in planes[:i]:
            lines.append(plane_intersection(plane1, plane2))

    tetrahedron_vertices = project_and_rotate(np.eye(4), angle=angle).T
    intersection_points = []
    for line in lines:
        intersection_points.append(line_tetrahedron_intersection(line, tetrahedron_vertices))

    print(intersection_points)

    for p in intersection_points:
        if len(p) > 0:
            p0, p1 = p
            x_values, y_values, z_values = zip(p0, p1)
            line = go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='lines',
                line=dict(
                    color=color,
                    width=linewidth
                )
            )
            fig.add_trace(line)

    return fig, intersection_points

def plot_normal(offset, normal, limits=[-1,1]):

    # plot the normal space at mu
    pass


def plot_occ_spaces(envs, angle=0.0, colors=None, labels=None):

    # Define the plane normal, plane point, and tetrahedron vertices
    colors = np.full(len(envs), 'darkgray') if colors is None else colors
    labels = np.full(len(envs), None) if labels is None else labels

    planes = []
    fig, plane = plot_occ_space(envs[0], angle=angle, color=colors[0], name=labels[0])
    planes.append(plane)
    for i, env in enumerate(envs[1:]):
        fig, plane = plot_occ_space(env, fig=fig, angle=angle, color=colors[i + 1], name=labels[i+1])
        planes.append(plane)
    return fig, planes


# def plot_reward_spaces()

def plot_policy_kl_ball(env, mu_c, beta, radius, resolution=100, limits=(-1.0,1.0)):

    # Define new coordinates on occupancy measure set
    U = orthogonal_complement(env.A_matrix())
    bu, bv = U.T[0], U.T[1]

    def project(mu):
        return U.T @ (mu - mu_c)

    def reconstruct(z):
        return mu_c + U @ z

    # Create a mesh within the simplex
    resolution = 100
    u_min, u_max = limits[0], limits[1]
    uu, vv = np.meshgrid(np.linspace(-u_min, u_max, resolution), np.linspace(-u_min, u_max, resolution))

    # Define f so that the ball is the level set f(mu)=0.
    def f(mu):
        if np.all(mu >= 0) & np.all(mu <= 1):
            pi_c = env.occ2policy(mu_c)
            pi = env.occ2policy(mu)
            nu = env.stateocc(mu)
            policy_kl = np.where(mu == 0, 0, np.where(mu_c == 0, float('inf'), mu * np.log(mu / mu_c)))
            return beta * np.sum(policy_kl) - radius
        else:
            return float('inf')








