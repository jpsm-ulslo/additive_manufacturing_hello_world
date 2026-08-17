import pyvista as pv
import trimesh

mesh = pv.read("model.obj")
meshV = trimesh.load("model.obj")
boundsV = meshV.bounds
xmin, ymin, zmin = boundsV[0]
xmax, ymax, zmax = boundsV[1]

print(boundsV)

#meshV = trimesh.load("szuflada.stl")
#meshV.export("model.obj")

plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightgrey")
plotter.add_axes()
plotter.show()