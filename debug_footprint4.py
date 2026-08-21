import sys
import tempfile

import cadquery as cq
import pyvista as pv
import numpy as np
import cv2

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from matplotlib.patches import Circle


PCB_LAYER_PERCENT = 0.15
CORNER_MM = 4

LIP_HEIGHT = 2.0
LIP_WIDTH  = 1.0


def fit_circle(points):

    x = points[:,0]
    y = points[:,1]

    x_m = np.mean(x)
    y_m = np.mean(y)

    u = x - x_m
    v = y - y_m

    Suu  = np.sum(u*u)
    Svv  = np.sum(v*v)
    Suv  = np.sum(u*v)

    Suuu = np.sum(u*u*u)
    Svvv = np.sum(v*v*v)

    Suuv = np.sum(u*u*v)
    Suvv = np.sum(u*v*v)

    A = np.array([
        [Suu, Suv],
        [Suv, Svv]
    ])

    B = np.array([
        (Suuu + Suvv)/2.0,
        (Svvv + Suuv)/2.0
    ])

    uc, vc = np.linalg.solve(A, B)

    xc = x_m + uc
    yc = y_m + vc

    r = np.mean(
        np.sqrt(
            (x-xc)**2 +
            (y-yc)**2
        )
    )

    return xc, yc, r

def refine_hole_local(
    xy,
    hole,
    search_factor=1.5
):

    cx, cy, r = hole

    search = r * search_factor

    pts = xy[

        (xy[:,0] >= cx-search) &
        (xy[:,0] <= cx+search) &

        (xy[:,1] >= cy-search) &
        (xy[:,1] <= cy+search)

    ]

    if len(pts) < 20:
        return hole

    dist = np.sqrt(
        (pts[:,0]-cx)**2 +
        (pts[:,1]-cy)**2
    )

    ring = pts[

        (dist >= r*0.6) &
        (dist <= r*1.4)

    ]

    if len(ring) < 12:
        return hole

    try:

        xc, yc, rr = fit_circle(ring)

        return (
            xc,
            yc,
            rr
        )

    except Exception:

        return hole
# --------------------------------------------------
# STEP -> MESH
# --------------------------------------------------

def load_mesh(stepfile):

    part = cq.importers.importStep(stepfile)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".stl",
        delete=False
    )

    cq.exporters.export(
        part,
        tmp.name
    )

    return pv.read(tmp.name)


# --------------------------------------------------
# PCB LAYER
# --------------------------------------------------

def pcb_points(mesh):

    z = mesh.points[:, 2]

    zmin = z.min()
    zmax = z.max()

    limit = zmin + (
        zmax - zmin
    ) * PCB_LAYER_PERCENT

    return mesh.points[
        z <= limit
    ]


# --------------------------------------------------
# CONTOUR
# --------------------------------------------------

def contour(points):

    xy = points[:, :2]

    hull = ConvexHull(xy)

    c = xy[hull.vertices]

    c = np.vstack(
        [c, c[0]]
    )

    return c


# --------------------------------------------------
# POINTS -> IMAGE
# --------------------------------------------------

def points_to_image(xy, scale=12):

    minx = np.min(xy[:, 0])
    maxx = np.max(xy[:, 0])

    miny = np.min(xy[:, 1])
    maxy = np.max(xy[:, 1])

    padding = 20

    w = int((maxx - minx) * scale) + padding * 2
    h = int((maxy - miny) * scale) + padding * 2

    img = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    for x, y in xy:

        px = int(
            (x - minx) * scale
        ) + padding

        py = int(
            (y - miny) * scale
        ) + padding

        cv2.circle(
            img,
            (px, py),
            1,
            255,
            -1
        )

    return (
        img,
        minx,
        miny,
        scale,
        padding
    )


# --------------------------------------------------
# HOUGH IN CORNERS
# --------------------------------------------------

def detect_corner_holes_hough(
        xy,
        contour_pts,
        corner_mm=8):

    (
        img,
        minx,
        miny,
        scale,
        padding
    ) = points_to_image(xy)

    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    mincx = np.min(contour_pts[:, 0])
    maxcx = np.max(contour_pts[:, 0])

    mincy = np.min(contour_pts[:, 1])
    maxcy = np.max(contour_pts[:, 1])

    corners = [

        ("TL", mincx, maxcy),
        ("TR", maxcx, maxcy),

        ("BL", mincx, mincy),
        ("BR", maxcx, mincy)

    ]

    holes = []

    for name, cx, cy in corners:

        px = int(
            (cx - minx) * scale
        ) + padding

        py = int(
            (cy - miny) * scale
        ) + padding

        win = int(
            corner_mm * scale
        )

        x0 = max(0, px - win)
        x1 = min(img.shape[1], px + win)

        y0 = max(0, py - win)
        y1 = min(img.shape[0], py + win)

        roi = img[
            y0:y1,
            x0:x1
        ]

        circles = cv2.HoughCircles(

            roi,

            cv2.HOUGH_GRADIENT,

            dp=1,

            minDist=20,

            param1=50,

            param2=10,

            minRadius=4,

            maxRadius=40

        )

        if circles is None:
            continue

        circles = np.round(
            circles[0]
        ).astype(int)

        circles = sorted(
            circles,
            key=lambda c: c[2],
            reverse=True
        )

        xx, yy, rr = circles[0]

        world_x = (
            x0 + xx - padding
        ) / scale + minx

        world_y = (
            y0 + yy - padding
        ) / scale + miny

        holes.append(
            (
                world_x,
                world_y,
                rr / scale
            )
        )

    return holes


# --------------------------------------------------
# LOAD
# --------------------------------------------------

mesh1 = load_mesh(sys.argv[1])

mesh1.rotate_z(
    90,
    inplace=True
)

mesh1.rotate_y(
    90,
    inplace=True
)

#mesh2 = load_mesh(sys.argv[2])

bbox1 = mesh1.bounds
#bbox2 = mesh2.bounds

width1 = bbox1[1] - bbox1[0]
#width2 = bbox2[1] - bbox2[0]

gap = width1
# gap = max(
#     width1#,
#     #width2
# ) * 0.2

# mesh2.translate(
#     [
#         width1 + gap,
#         0,
#         0
#     ],
#     inplace=True
# )

# --------------------------------------------------
# POINTS
# --------------------------------------------------

all_xy1 = mesh1.points[:, :2]
#all_xy2 = mesh2.points[:, :2]

pcb1 = pcb_points(mesh1)
#pcb2 = pcb_points(mesh2)

pcb_xy1 = pcb1[:, :2]
#pcb_xy2 = pcb2[:, :2]

contour1 = contour(pcb1)
#contour2 = contour(pcb2)

# holes1 = detect_corner_holes_hough(
#     pcb_xy1,
#     contour1,
#     corner_mm=CORNER_MM
# )

# holes2 = detect_corner_holes_hough(
#     pcb_xy2,
#     contour2,
#     corner_mm=CORNER_MM
# )

#-----------------------------------------------
holes1_raw = detect_corner_holes_hough(
    pcb_xy1,
    contour1,
    corner_mm=CORNER_MM
)

# holes2_raw = detect_corner_holes_hough(
#     pcb_xy2,
#     contour2,
#     corner_mm=CORNER_MM
# )

holes1 = [
    refine_hole_local(
        pcb_xy1,
        h
    )
    for h in holes1_raw
]

# holes2 = [
#     refine_hole_local(
#         pcb_xy2,
#         h
#     )
#     for h in holes2_raw
# ]
#----------------------------------------------- 

print("\nLM2596 holes")

for h in holes1:
    print(h)

print("\nRelay holes")

# for h in holes2:
#     print(h)

# # --------------------------------------------------
# # FIGURE
# # --------------------------------------------------

# fig, axs = plt.subplots(
#     2,
#     2,
#     figsize=(14, 12)
# )

# # ------------------------------------------

# ax = axs[0, 0]

# ax.scatter(
#     all_xy1[:,0],
#     all_xy1[:,1],
#     s=0.2,
#     c="blue",
#     alpha=0.5
# )

# ax.scatter(
#     all_xy2[:,0],
#     all_xy2[:,1],
#     s=0.2,
#     c="green",
#     alpha=0.5
# )

# ax.set_title(
#     "Full projection"
# )

# ax.set_aspect("equal")

# # ------------------------------------------

# ax = axs[0, 1]

# ax.scatter(
#     pcb_xy1[:,0],
#     pcb_xy1[:,1],
#     s=1,
#     c="blue"
# )

# ax.scatter(
#     pcb_xy2[:,0],
#     pcb_xy2[:,1],
#     s=1,
#     c="green"
# )

# ax.set_title(
#     "PCB layer only"
# )

# ax.set_aspect("equal")

# # ------------------------------------------

# ax = axs[1, 0]

# ax.scatter(
#     pcb_xy1[:,0],
#     pcb_xy1[:,1],
#     s=1,
#     c="blue"
# )

# ax.scatter(
#     pcb_xy2[:,0],
#     pcb_xy2[:,1],
#     s=1,
#     c="green"
# )

# ax.plot(
#     contour1[:,0],
#     contour1[:,1],
#     color="red",
#     linewidth=2
# )

# ax.plot(
#     contour2[:,0],
#     contour2[:,1],
#     color="orange",
#     linewidth=2
# )

# # Hough original
# for x,y,r in holes1_raw:

#     ax.add_patch(
#         Circle(
#             (x,y),
#             r,
#             fill=False,
#             color="magenta",
#             linewidth=1
#         )
#     )

# # Refinado
# for x,y,r in holes1:

#     ax.add_patch(
#         Circle(
#             (x,y),
#             r,
#             fill=False,
#             color="lime",
#             linewidth=3
#         )
#     )


# # Hough original
# for x,y,r in holes2_raw:

#     ax.add_patch(
#         Circle(
#             (x,y),
#             r,
#             fill=False,
#             color="magenta",
#             linewidth=1
#         )
#     )

# # Refinado
# for x,y,r in holes2:

#     ax.add_patch(
#         Circle(
#             (x,y),
#             r,
#             fill=False,
#             color="lime",
#             linewidth=3
#         )
#     )


# ax.set_title(
#     "Detected footprint + peg hypothesis"
# )

# ax.set_aspect("equal")

# # ------------------------------------------

# ax = axs[1, 1]

# ax.scatter(
#     pcb_xy1[:,0],
#     pcb_xy1[:,1],
#     s=0.5,
#     c="lightblue"
# )

# ax.scatter(
#     pcb_xy2[:,0],
#     pcb_xy2[:,1],
#     s=0.5,
#     c="lightgreen"
# )

# ax.plot(
#     contour1[:,0],
#     contour1[:,1],
#     linewidth=3,
#     color="red"
# )

# ax.plot(
#     contour2[:,0],
#     contour2[:,1],
#     linewidth=3,
#     color="orange"
# )

# for x,y,r in holes1:

#     ax.add_patch(
#         Circle(
#             (x,y),
#             r,
#             fill=False,
#             color="magenta",
#             linewidth=3
#         )
#     )

# for x,y,r in holes2:

#     ax.add_patch(
#         Circle(
#             (x,y),
#             r,
#             fill=False,
#             color="cyan",
#             linewidth=3
#         )
#     )

# ax.set_title(
#     "Proposed holder footprint + pegs"
# )

# ax.set_aspect("equal")

# plt.tight_layout()
# plt.show()

# # --------------------------------------------------
# # ORIGINAL 3D VIEW
# # --------------------------------------------------

# p = pv.Plotter()

# p.add_mesh(
#     mesh1,
#     color="blue",
#     opacity=0.7
# )

# p.add_mesh(
#     mesh2,
#     color="green",
#     opacity=0.7
# )

# p.add_axes()
# p.show_grid()

# p.show()

# --------------------------------------------------
# PYVISTA TOP VIEW
# --------------------------------------------------

p = pv.Plotter()

#
# PCB points
#

pcb1_cloud = pv.PolyData(
    pcb1
)

# pcb2_cloud = pv.PolyData(
#     pcb2
# )

p.add_mesh(
    pcb1_cloud,
    color="blue",
    point_size=4,
    render_points_as_spheres=True
)

# p.add_mesh(
#     pcb2_cloud,
#     color="green",
#     point_size=4,
#     render_points_as_spheres=True
# )

#
# Contours
#

def contour_polydata(contour):

    pts = np.c_[

        contour[:,0],
        contour[:,1],
        np.zeros(
            len(contour)
        )

    ]

    poly = pv.PolyData(pts)

    cells = []

    for i in range(
        len(pts)-1
    ):

        cells.extend(
            [
                2,
                i,
                i+1
            ]
        )

    poly.lines = np.array(
        cells
    )

    return poly

p.add_mesh(
    contour_polydata(contour1),
    color="red",
    line_width=5
)

# p.add_mesh(
#     contour_polydata(contour2),
#     color="orange",
#     line_width=5
# )

#
# Hough circles
#

for x,y,r in holes1_raw:

    p.add_mesh(

        pv.Circle(
            radius=r,
            resolution=64
        ).translate(
            (
                x,
                y,
                0
            ),
            inplace=False
        ),

        color="magenta",
        line_width=2

    )

# for x,y,r in holes2_raw:

#     p.add_mesh(

#         pv.Circle(
#             radius=r,
#             resolution=64
#         ).translate(
#             (
#                 x,
#                 y,
#                 0
#             ),
#             inplace=False
#         ),

#         color="cyan",
#         line_width=2

#     )

#
# Refined circles
#

for x,y,r in holes1:

    p.add_mesh(

        pv.Circle(
            radius=r,
            resolution=64
        ).translate(
            (
                x,
                y,
                0
            ),
            inplace=False
        ),

        color="lime",
        line_width=4

    )

    p.add_mesh(

        pv.Sphere(
            radius=0.4,
            center=(x,y,0)
        ),

        color="yellow"

    )

# for x,y,r in holes2:

#     p.add_mesh(

#         pv.Circle(
#             radius=r,
#             resolution=64
#         ).translate(
#             (
#                 x,
#                 y,
#                 0
#             ),
#             inplace=False
#         ),

#         color="yellow",
#         line_width=4

#     )

#     p.add_mesh(

#         pv.Sphere(
#             radius=0.4,
#             center=(x,y,0)
#         ),

#         color="red"

#     )

#
# Top-down camera
#

p.camera_position = "xy"

p.add_axes()

p.show_grid()

p.show()


BASE_THICKNESS = 2.4

CLEARANCE = 0.5

WALL_THICKNESS = 2.0
WALL_HEIGHT = 2.0

PEG_HEIGHT = 5.0

BASE_MARGIN = 5.0

# ==================================================
# BUILD HOLDER
# ==================================================

def contour_bounds(contour):

    return (

        np.min(contour[:,0]),
        np.min(contour[:,1]),

        np.max(contour[:,0]),
        np.max(contour[:,1])

    )

xmin1,ymin1,xmax1,ymax1 = \
    contour_bounds(contour1)

# xmin2,ymin2,xmax2,ymax2 = \
#     contour_bounds(contour2)
global_minx = xmin1 - BASE_MARGIN
# global_minx = min(
#     xmin1,
#     # xmin2
# ) - BASE_MARGIN

global_maxx = xmax1 + BASE_MARGIN
# global_maxx = max(
#     xmax1,
#     #xmax2
# ) + BASE_MARGIN

global_miny = ymin1 - BASE_MARGIN
# global_miny = min(
#     ymin1,
#     #ymin2
# ) - BASE_MARGIN

global_maxy = ymax1 + BASE_MARGIN
# global_maxy = max(
#     ymax1,
#     #ymax2
# ) + BASE_MARGIN


# holder = (

#     cq.Workplane("XY")

#     .rect(

#         global_maxx-global_minx,

#         global_maxy-global_miny

#     )

#     .extrude(
#         BASE_THICKNESS
#     )

# )

base_cx = (
    global_minx + global_maxx
) / 2

base_cy = (
    global_miny + global_maxy
) / 2

# holder = (

#     cq.Workplane("XY")

#     .center(
#         base_cx,
#         base_cy
#     )

#     .rect(
#         global_maxx-global_minx,
#         global_maxy-global_miny
#     )

#     .extrude(
#         BASE_THICKNESS
#     )

# )

BASE_CORNER_RADIUS = 4.0

holder = (

    cq.Workplane("XY")

    .center(
        base_cx,
        base_cy
    )

    .rect(
        global_maxx-global_minx,
        global_maxy-global_miny
    )

    .extrude(
        BASE_THICKNESS
    )

    .edges("|Z")

    .fillet(
        BASE_CORNER_RADIUS
    )

)

BOX_WALL_HEIGHT = 25.0

box_outer = (

    cq.Workplane("XY")

    .center(
        base_cx,
        base_cy
    )

    .rect(
        global_maxx-global_minx,
        global_maxy-global_miny
    )

    .extrude(
        BOX_WALL_HEIGHT + BASE_THICKNESS
    )

)

box_inner = (

    cq.Workplane("XY")

    .center(
        base_cx,
        base_cy
    )

    .workplane(
        offset=BASE_THICKNESS
    )

    .rect(

        (global_maxx-global_minx)
        - 2*WALL_THICKNESS,

        (global_maxy-global_miny)
        - 2*WALL_THICKNESS

    )

    .extrude(
        BOX_WALL_HEIGHT + 1
    )

)


def wall_from_rect(
    xmin,
    ymin,
    xmax,
    ymax
):

    outer = (

        cq.Workplane("XY")

        .center(
            (xmin+xmax)/2,
            (ymin+ymax)/2
        )

        .rect(
            (xmax-xmin)
            + 2*(CLEARANCE+WALL_THICKNESS),

            (ymax-ymin)
            + 2*(CLEARANCE+WALL_THICKNESS)
        )

        .extrude(
            WALL_HEIGHT
        )
        .faces(">Z")

        .chamfer(0.1)

    )

    inner = (

        cq.Workplane("XY")

        .center(
            (xmin+xmax)/2,
            (ymin+ymax)/2
        )

        .rect(

            (xmax-xmin)
            + 2*CLEARANCE,

            (ymax-ymin)
            + 2*CLEARANCE

        )

        .extrude(
            WALL_HEIGHT+1
        )
        .faces(">Z")

        .chamfer(0.1)


    )

    return outer.cut(
        inner
    )


holder = box_outer.cut(
    box_inner
)

holder = (

    holder

    .edges("|Z")

    .fillet(2.0)

)




lip = (

    cq.Workplane("XY")

    .center(
        base_cx,
        base_cy
    )

    .workplane(
        offset=
        BASE_THICKNESS
        +
        BOX_WALL_HEIGHT
        -
        LIP_HEIGHT
    )

    .rect(

        (global_maxx-global_minx)
        - 2*WALL_THICKNESS,

        (global_maxy-global_miny)
        - 2*WALL_THICKNESS

    )

    .extrude(
        LIP_HEIGHT
    )

)

inner_lip = (

    cq.Workplane("XY")

    .center(
        base_cx,
        base_cy
    )

    .workplane(
        offset=
        BASE_THICKNESS
        +
        BOX_WALL_HEIGHT
        -
        LIP_HEIGHT
    )

    .rect(

        (global_maxx-global_minx)
        - 2*WALL_THICKNESS
        - 2*LIP_WIDTH,

        (global_maxy-global_miny)
        - 2*WALL_THICKNESS
        - 2*LIP_WIDTH

    )

    .extrude(
        LIP_HEIGHT + 0.1
    )

)

lip = lip.cut(
    inner_lip
)


# holder = holder.union(
#     lip
# )

# holder = holder.union(

#     wall_from_rect(
#         xmin1,
#         ymin1,
#         xmax1,
#         ymax1
#     ).translate(
#         (0,0,BASE_THICKNESS)
#     )

# )

# holder = holder.union(

#     wall_from_rect(
#         xmin2,
#         ymin2,
#         xmax2,
#         ymax2
#     ).translate(
#         (0,0,BASE_THICKNESS)
#     )

# )

# for x,y,r in holes1:

#     peg = (

#         cq.Workplane("XY")

#         .center(x,y)

#         .circle(
#             max(
#                 0.8,
#                 r-0.15
#             )
#         )

#         .extrude(
#             PEG_HEIGHT
#         )

#         .faces(">Z")

#         .chamfer(0.3)

#     )

#     holder = holder.union(

#         peg.translate(
#             (
#                 0,
#                 0,
#                 BASE_THICKNESS
#             )
#         )

#     )

for x, y, r in holes1:

    hole_diameter = 2 * r

    peg_diameter = max(
        1.6,
        hole_diameter - 0.30
    )

    peg = (

        cq.Workplane("XY")

        .center(x, y)

        .circle(
            peg_diameter / 2
        )

        .extrude(
            PEG_HEIGHT
        )

        .faces(">Z")

        .chamfer(0.6)

    )

    holder = holder.union(

        peg.translate(
            (
                0,
                0,
                BASE_THICKNESS
            )
        )

    )

#---------------------------
# for x,y,r in holes2:

#     peg = (

#         cq.Workplane("XY")

#         .center(x,y)

#         .circle(
#             max(
#                 0.8,
#                 r-0.15
#             )
#         )

#         .extrude(
#             PEG_HEIGHT
#         )

#         .faces(">Z")

#         .chamfer(0.3)

#     )

#     holder = holder.union(

#         peg.translate(
#             (
#                 0,
#                 0,
#                 BASE_THICKNESS
#             )
#         )

#     )

print(
    "box width",
    global_maxx-global_minx
)

print(
    "box depth",
    global_maxy-global_miny
)

print(
    "box height",
    BOX_WALL_HEIGHT + BASE_THICKNESS
)

cq.exporters.export(
    holder,
    "holder.stl"
)

cq.exporters.export(
    holder,
    "holder.step"
)

# holder_mesh = pv.read(
#     "holder.stl"
# )

# final = pv.Plotter()

# final.add_mesh(
#     holder_mesh,
#     color="lightgray"
# )

# final.add_mesh(
#     mesh1,
#     color="blue",
#     opacity=0.8
# )

# final.add_mesh(
#     mesh2,
#     color="green",
#     opacity=0.8
# )

holder_mesh = pv.read(
    "holder.stl"
)

#
# assentar os módulos na base
#

mesh1_bottom = mesh1.bounds[4]
# mesh2_bottom = mesh2.bounds[4]

mesh1.translate(
    (
        0,
        0,
        BASE_THICKNESS - mesh1_bottom
    ),
    inplace=True
)

# mesh2.translate(
#     (
#         0,
#         0,
#         BASE_THICKNESS - mesh2_bottom
#     ),
#     inplace=True
# )

final = pv.Plotter()

final.add_mesh(
    holder_mesh,
    color="lightgray",
    opacity=1.0
)

final.add_mesh(
    mesh1,
    color="blue",
    opacity=0.7
)

# final.add_mesh(
#     mesh2,
#     color="green",
#     opacity=0.7
# )

# final.view_xy()

# final.enable_parallel_projection()

final.view_isometric()

final.add_axes()

final.show_grid()

print(
    "Base bounds:",
    holder_mesh.bounds
)

print(
    "LM2596 bounds:",
    mesh1.bounds
)

# print(
#     "Relay bounds:",
#     mesh2.bounds
# )

final.show()