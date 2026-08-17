import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from matplotlib.textpath import TextPath
from svgpathtools import svg2paths
from shapely.geometry import Polygon, box


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ESPESSURA_PLACA = 5.0

PROF_ICON = 2.0
PROF_TITULO = 1.5
PROF_SUBTITULO = 1.5

ARQUIVO_3MF = "placa_multicolor.3mf"

# Cores usadas no 3MF
COR_VERDE = [0, 180, 0, 255]
COR_BRANCO = [255, 255, 255, 255]


# ============================================================
# AUXILIAR
# ============================================================

def pts_from_segments(segments, samples=40):

    pts = []

    for seg in segments:

        for t in np.linspace(
            0,
            1,
            samples,
            endpoint=False
        ):

            p = seg.point(t)

            pts.append(
                (
                    float(p.real),
                    float(p.imag)
                )
            )

    return pts


def extrude_poly(poly, altura):

    poly = poly.buffer(0)

    if poly.is_empty:
        raise RuntimeError(
            "Polígono vazio durante extrusão"
        )

    if poly.geom_type == "Polygon":

        return trimesh.creation.extrude_polygon(
            poly,
            altura
        )

    meshes = []

    for p in poly.geoms:

        if p.is_empty:
            continue

        meshes.append(
            trimesh.creation.extrude_polygon(
                p,
                altura
            )
        )

    if not meshes:

        raise RuntimeError(
            "Nenhum polígono válido para extrusão"
        )

    return trimesh.util.concatenate(
        meshes
    )


def aplicar_cor(mesh, cor):

    """
    Define cor visual por face.

    Isto não altera a geometria.
    Serve para o 3MF transportar a aparência
    do objeto quando o software de destino
    respeita vertex/face colors.
    """

    mesh.visual.face_colors = np.tile(
        np.array(cor, dtype=np.uint8),
        (len(mesh.faces), 1)
    )


# ============================================================
# PLACA
# ============================================================

print()
print("========================================")
print("CRIANDO PLACA")
print("========================================")

tree = ET.parse(
    "0_placa.svg"
)

root = tree.getroot()

placa_poly = None

for elem in root.iter():

    tag = elem.tag.split("}")[-1]

    if tag != "rect":
        continue

    style = elem.attrib.get(
        "style",
        ""
    ).lower()

    if "fill:#000000" not in style:
        continue

    x = float(
        elem.attrib["x"]
    )

    y = float(
        elem.attrib["y"]
    )

    w = float(
        elem.attrib["width"]
    )

    h = float(
        elem.attrib["height"]
    )

    rx = float(
        elem.attrib.get(
            "rx",
            0
        )
    )

    placa_poly = box(
        x,
        y,
        x + w,
        y + h
    )

    if rx > 0:

        placa_poly = (
            placa_poly
            .buffer(-rx)
            .buffer(rx)
        )

    break


if placa_poly is None:

    raise RuntimeError(
        "Placa não encontrada"
    )


placa_mesh = trimesh.creation.extrude_polygon(
    placa_poly,
    ESPESSURA_PLACA
)

aplicar_cor(
    placa_mesh,
    COR_VERDE
)

print("PLACA OK")
print(
    "Bounds:",
    placa_mesh.bounds
)


# ============================================================
# ICON
# ============================================================

print()
print("========================================")
print("CRIANDO ICON")
print("========================================")

paths, attrs = svg2paths(
    "0_icon.svg"
)

path_icon = None

for path, attr in zip(
    paths,
    attrs
):

    if attr.get("id") == "path3":

        path_icon = path
        break


if path_icon is None:

    raise RuntimeError(
        "path3 do icon não encontrado"
    )


outer_pts = pts_from_segments(
    path_icon[:47]
)

hole_left = pts_from_segments(
    path_icon[47:74]
)

hole_right = pts_from_segments(
    path_icon[74:100]
)


icon_poly = Polygon(
    shell=outer_pts,
    holes=[
        hole_left,
        hole_right
    ]
).buffer(0)


# ============================================================
# ALTO RELEVO
#
# A placa termina em Z = 5.0
# O icon começa em Z = 5.0
# O icon termina em Z = 7.0
# ============================================================

icon_mesh = trimesh.creation.extrude_polygon(
    icon_poly,
    PROF_ICON
)

icon_mesh.apply_translation(
    [
        0,
        0,
        ESPESSURA_PLACA
    ]
)

aplicar_cor(
    icon_mesh,
    COR_BRANCO
)

print("ICON OK")
print(
    "Bounds:",
    icon_mesh.bounds
)


# ============================================================
# TITULO
# ============================================================

print()
print("========================================")
print("CRIANDO TITULO")
print("========================================")

paths, attrs = svg2paths(
    "0_titulo.svg"
)

path_titulo = None

for path, attr in zip(
    paths,
    attrs
):

    if attr.get("id") == "path3":

        path_titulo = path
        break


if path_titulo is None:

    raise RuntimeError(
        "path3 do titulo não encontrado"
    )


def ciclo_pts(a, b):

    return pts_from_segments(
        path_titulo[a:b + 1]
    )


CICLO_G = (0, 42)
CICLO_A = (44, 71)
CICLO_M = (73, 115)

CICLO_C = (117, 143)

CICLO_O_OUTER = (145, 159)
CICLO_O_HOLE = (161, 171)

CICLO_R = (173, 188)
CICLO_E = (190, 219)


titulo_meshes = []


# ------------------------------------------------------------
# C
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            ciclo_pts(
                *CICLO_C
            )
        ),
        PROF_TITULO
    )
)


# ------------------------------------------------------------
# O
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            shell=ciclo_pts(
                *CICLO_O_OUTER
            ),
            holes=[
                ciclo_pts(
                    *CICLO_O_HOLE
                )
            ]
        ),
        PROF_TITULO
    )
)


# ------------------------------------------------------------
# R
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            ciclo_pts(
                *CICLO_R
            )
        ),
        PROF_TITULO
    )
)


# ------------------------------------------------------------
# A
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            ciclo_pts(
                *CICLO_A
            )
        ),
        PROF_TITULO
    )
)


# ------------------------------------------------------------
# G
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            ciclo_pts(
                *CICLO_G
            )
        ),
        PROF_TITULO
    )
)


# ------------------------------------------------------------
# E
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            ciclo_pts(
                *CICLO_E
            )
        ),
        PROF_TITULO
    )
)


# ------------------------------------------------------------
# M
# ------------------------------------------------------------

titulo_meshes.append(
    extrude_poly(
        Polygon(
            ciclo_pts(
                *CICLO_M
            )
        ),
        PROF_TITULO
    )
)


titulo_mesh = trimesh.util.concatenate(
    titulo_meshes
)


# ============================================================
# POSIÇÃO Z
#
# Começa no topo da placa:
#
# Z = 5.0
# Z = 6.5
# ============================================================

titulo_mesh.apply_translation(
    [
        0,
        0,
        ESPESSURA_PLACA
    ]
)


# ============================================================
# ESPELHAMENTO HORIZONTAL
# ============================================================

cx = (
    placa_poly.bounds[0]
    +
    placa_poly.bounds[2]
) / 2.0


M = np.array([
    [-1, 0, 0, 2 * cx],
    [ 0, 1, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1]
])


icon_mesh.apply_transform(
    M
)

titulo_mesh.apply_transform(
    M
)


aplicar_cor(
    titulo_mesh,
    COR_BRANCO
)


print("TITULO OK")
print(
    "Bounds:",
    titulo_mesh.bounds
)


# ============================================================
# SUBTITULO
# ============================================================

print()
print("========================================")
print("CRIANDO SUBTITULO")
print("========================================")


tree = ET.parse(
    "0_subtitulo.svg"
)

root = tree.getroot()


SUB_X = None
SUB_Y = None
TAMANHO = None


for elem in root.iter():

    tag = elem.tag.split("}")[-1]

    if tag != "text":
        continue

    SUB_X = float(
        elem.attrib.get(
            "x",
            0
        )
    )

    SUB_Y = float(
        elem.attrib.get(
            "y",
            0
        )
    )

    style = elem.attrib.get(
        "style",
        ""
    )

    for item in style.split(";"):

        item = item.strip()

        if item.startswith(
            "font-size:"
        ):

            TAMANHO = float(
                item
                .replace(
                    "font-size:",
                    ""
                )
                .replace(
                    "px",
                    ""
                )
            )

            break

    break


if SUB_X is None:

    raise RuntimeError(
        "Texto do subtitulo não encontrado"
    )


if TAMANHO is None:

    raise RuntimeError(
        "font-size do subtitulo não encontrado"
    )


print()
print(
    "SUB_X =",
    SUB_X
)

print(
    "SUB_Y =",
    SUB_Y
)

print(
    "TAMANHO =",
    TAMANHO
)


# ============================================================
# TEXTO DO SUBTITULO
# ============================================================

TEXTO_SUB = [
    "Associação de Apoio",
    "a Crianças com Doença Cardíaca"
]


def construir_poligonos_com_holes(
    texto,
    tamanho
):

    tp = TextPath(
        (0, 0),
        texto,
        size=tamanho
    )

    raw = []

    for pts in tp.to_polygons():

        if len(pts) < 3:
            continue

        poly = Polygon(
            pts
        ).buffer(0)

        if poly.is_empty:
            continue

        if poly.area > 0:

            raw.append(
                poly
            )


    exteriores = []


    for p in raw:

        parent = None


        for q in raw:

            if p.equals(q):
                continue

            if q.contains(p):

                if (
                    parent is None
                    or
                    parent.area > q.area
                ):

                    parent = q


        if parent is None:

            holes = []


            for h in raw:

                if p.equals(h):
                    continue

                if p.contains(h):

                    holes.append(
                        list(
                            h.exterior.coords
                        )
                    )


            exteriores.append(
                Polygon(
                    shell=list(
                        p.exterior.coords
                    ),
                    holes=holes
                )
            )


    return exteriores


# ============================================================
# CONSTRUIR SUBTITULO
# ============================================================

subtitulo_meshes = []


for idx, linha in enumerate(
    TEXTO_SUB
):

    polys = construir_poligonos_com_holes(
        linha,
        TAMANHO
    )


    y = (
        SUB_Y
        -
        idx * TAMANHO * 1.4
    )


    linha_meshes = []


    for poly in polys:

        poly = poly.buffer(0)

        if poly.is_empty:
            continue


        mesh = trimesh.creation.extrude_polygon(
            poly,
            PROF_SUBTITULO
        )


        linha_meshes.append(
            mesh
        )


    if not linha_meshes:
        continue


    linha_mesh = trimesh.util.concatenate(
        linha_meshes
    )


    # --------------------------------------------------------
    # BBOX DA LINHA
    # --------------------------------------------------------

    bbox = TextPath(
        (0, 0),
        linha,
        size=TAMANHO
    ).get_extents()


    # --------------------------------------------------------
    # POSIÇÃO
    #
    # Z = 5.0
    #
    # Não usamos mais:
    #
    # ESPESSURA_PLACA - PROF_SUBTITULO
    #
    # porque agora é ALTO RELEVO.
    # --------------------------------------------------------

    linha_mesh.apply_translation(
        [
            SUB_X,
            y - bbox.ymin,
            ESPESSURA_PLACA
        ]
    )


    subtitulo_meshes.append(
        linha_mesh
    )


if not subtitulo_meshes:

    raise RuntimeError(
        "Nenhuma geometria do subtitulo foi criada"
    )


subtitulo_mesh = trimesh.util.concatenate(
    subtitulo_meshes
)


# ============================================================
# MESMAS TRANSFORMAÇÕES DO SCRIPT ORIGINAL
# ============================================================

subtitulo_mesh.apply_transform(
    M
)


# ------------------------------------------------------------
# CORREÇÃO DO ESPELHAMENTO VERTICAL
# ------------------------------------------------------------

bbox = subtitulo_mesh.bounds


cy = (
    subtitulo_mesh.bounds[0][1]
    +
    subtitulo_mesh.bounds[1][1]
) / 2.0


MY = np.array([
    [1,  0, 0, 0],
    [0, -1, 0, 2 * cy],
    [0,  0, 1, 0],
    [0,  0, 0, 1]
])


subtitulo_mesh.apply_transform(
    MY
)


aplicar_cor(
    subtitulo_mesh,
    COR_BRANCO
)


print("SUBTITULO OK")
print(
    "Bounds:",
    subtitulo_mesh.bounds
)


# ============================================================
# VERIFICAÇÃO DOS QUATRO CORPOS
# ============================================================

print()
print("========================================")
print("VERIFICAÇÃO")
print("========================================")


def mostrar_bounds(
    nome,
    mesh
):

    b = mesh.bounds

    print()
    print(nome)

    print(
        "X:",
        b[0][0],
        "->",
        b[1][0]
    )

    print(
        "Y:",
        b[0][1],
        "->",
        b[1][1]
    )

    print(
        "Z:",
        b[0][2],
        "->",
        b[1][2]
    )


mostrar_bounds(
    "PLACA",
    placa_mesh
)

mostrar_bounds(
    "ICON",
    icon_mesh
)

mostrar_bounds(
    "TITULO",
    titulo_mesh
)

mostrar_bounds(
    "SUBTITULO",
    subtitulo_mesh
)


# ============================================================
# CRIAR SCENE
# ============================================================

print()
print("========================================")
print("CRIANDO SCENE 3MF")
print("========================================")


scene = trimesh.Scene()


# ------------------------------------------------------------
# PLACA VERDE
# ------------------------------------------------------------

scene.add_geometry(
    placa_mesh,
    geom_name="PLACA_VERDE",
    node_name="PLACA_VERDE"
)


# ------------------------------------------------------------
# ICON BRANCO
# ------------------------------------------------------------

scene.add_geometry(
    icon_mesh,
    geom_name="ICON_BRANCO",
    node_name="ICON_BRANCO"
)


# ------------------------------------------------------------
# TITULO BRANCO
# ------------------------------------------------------------

scene.add_geometry(
    titulo_mesh,
    geom_name="TITULO_BRANCO",
    node_name="TITULO_BRANCO"
)


# ------------------------------------------------------------
# SUBTITULO BRANCO
# ------------------------------------------------------------

scene.add_geometry(
    subtitulo_mesh,
    geom_name="SUBTITULO_BRANCO",
    node_name="SUBTITULO_BRANCO"
)


# ============================================================
# EXPORTAR 3MF
# ============================================================

print()
print("========================================")
print("EXPORTANDO 3MF")
print("========================================")


try:

    scene.export(
        ARQUIVO_3MF,
        file_type="3mf"
    )

except Exception as e:

    print()
    print("ERRO AO EXPORTAR 3MF:")
    print(e)

    raise


# ============================================================
# EXPORTAR STLs INDIVIDUAIS PARA DEBUG
# ============================================================

placa_mesh.export(
    "debug_placa_verde.stl"
)

icon_mesh.export(
    "debug_icon_branco.stl"
)

titulo_mesh.export(
    "debug_titulo_branco.stl"
)

subtitulo_mesh.export(
    "debug_subtitulo_branco.stl"
)


# ============================================================
# FINAL
# ============================================================

print()
print("========================================")
print("CONCLUÍDO")
print("========================================")
print()

print(
    "3MF:",
    ARQUIVO_3MF
)

print()

print("Objetos dentro da Scene:")

for nome in scene.geometry.keys():

    print(
        " -",
        nome
    )

print()

print("CORES:")
print(" - PLACA      = VERDE")
print(" - ICON       = BRANCO")
print(" - TITULO     = BRANCO")
print(" - SUBTITULO  = BRANCO")

print()

print("ALTURAS:")

print(
    " - Placa:",
    f"0 -> {ESPESSURA_PLACA} mm"
)

print(
    " - Icon:",
    f"{ESPESSURA_PLACA} -> "
    f"{ESPESSURA_PLACA + PROF_ICON} mm"
)

print(
    " - Titulo:",
    f"{ESPESSURA_PLACA} -> "
    f"{ESPESSURA_PLACA + PROF_TITULO} mm"
)

print(
    " - Subtitulo:",
    f"{ESPESSURA_PLACA} -> "
    f"{ESPESSURA_PLACA + PROF_SUBTITULO} mm"
)

print()
print("FIM")
print()

input("ENTER...")
