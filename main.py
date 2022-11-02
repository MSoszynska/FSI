import os
import time

from fenics import (
    parameters,
    RectangleMesh,
    Point,
    BoundaryMesh,
    SubMesh,
    HDF5File,
    MPI,
    MeshFunction,
    cells,
    refine,
    VectorFunctionSpace,
    ALE,
    Expression,
    project,
)
from mshr import Rectangle, Circle, generate_mesh, Polygon
from parameters import Parameters
from spaces import Inner_boundary, Space
from time_structure import TimeLine, split
from time_stepping import time_stepping
from forms import (
    bilinear_form_fluid,
    functional_fluid,
    bilinear_form_solid,
    functional_solid,
    functional_solid_interface,
    bilinear_form_fluid_adjoint,
    functional_fluid_adjoint,
    bilinear_form_solid_adjoint,
    functional_solid_adjoint,
    Problem,
)
from relaxation import relaxation
from shooting import shooting
from compute_residuals import compute_residuals
from refine import refine_time
from coupling import (
    interface_coordinates,
    interface_coordinates_transfer_table,
    extension_interface_coordinates_transfer_table,
)
from error_estimate import goal_functional_solid

parameters["allow_extrapolation"] = True
param = Parameters()

# Create meshes
rectangle_width = 0.65
trapezoid_width = 0.94914
trapezoid_height = 0.1419861277198325
channel = Polygon(
    [
        Point(rectangle_width, 0.41),
        Point(0.0, 0.41),
        Point(0.0, 0.0),
        Point(rectangle_width, 0.0),
        Point(
            rectangle_width + trapezoid_width, 0.205 - trapezoid_height / 2.0
        ),
        Point(
            rectangle_width + trapezoid_width, 0.205 + trapezoid_height / 2.0
        ),
    ]
)
cylinder = Circle(Point(0.2, 0.2), 0.05, 25)
elastic_structure = Rectangle(Point(0.2, 0.19), Point(0.6, 0.21))
domain_fluid = channel - cylinder - elastic_structure
domain_solid = elastic_structure - cylinder
mesh_fluid = generate_mesh(domain_fluid, 35)
mesh_solid = generate_mesh(domain_solid, 30)

# Refine fluid mesh
for i in range(2):
    markers = MeshFunction("bool", mesh_fluid, mesh_fluid.topology().dim())
    markers.set_all(False)
    for cell in cells(mesh_fluid):
        if i == 0:
            if 0.17 < cell.midpoint().y() and cell.midpoint().y() < 0.23:
                if 0.26 < cell.midpoint().x() and cell.midpoint().x() < 0.62:
                    markers[cell.index()] = True
            if cell.midpoint().distance(Point(0.2, 0.2)) < 0.07:
                if cell.midpoint().distance(Point(0.25, 0.2)) > 0.02:
                    markers[cell.index()] = True
        else:
            if 0.18 < cell.midpoint().y() and cell.midpoint().y() < 0.22:
                if 0.27 < cell.midpoint().x() and cell.midpoint().x() < 0.61:
                    markers[cell.index()] = True
    mesh_fluid = refine(mesh_fluid, markers)

# Move fluid mesh
displacement_space = VectorFunctionSpace(mesh_fluid, "CG", 1)
displacement = Expression(
    (
        "x[0] > rectangle_width ? (x[0] - rectangle_width) * (x[0] - rectangle_width)  : 0",
        "x[0] > rectangle_width ? 0.41 * trapezoid_width/ ((trapezoid_height - 0.41)  "
        "* x[0] + 0.41 * rectangle_width + 0.41 * trapezoid_width - trapezoid_height * rectangle_width) * "
        "(x[1] - 0.205) + 0.205 - x[1] : 0.0",
    ),
    rectangle_width=rectangle_width,
    trapezoid_width=trapezoid_width,
    trapezoid_height=trapezoid_height,
    degree=1,
)
displacement_function = project(displacement, displacement_space)
ALE.move(mesh_fluid, displacement_function)

# Define interface mesh
boundary_mesh = BoundaryMesh(mesh_fluid, "exterior")
inner_boundary = Inner_boundary()
mesh_interface = SubMesh(boundary_mesh, inner_boundary)

# Create function spaces
fluid_dimension = [2, 2, 1]
fluid_degree = [2, 2, 1]
solid_dimension = [2, 2]
solid_degree = [2, 2]
fluid = Space(mesh_fluid, fluid_dimension, fluid_degree, "fluid")
solid = Space(mesh_solid, solid_dimension, solid_degree, "solid")
interface = Space(mesh_interface, solid_dimension, solid_degree, "interface")

# Create tables of coordinates
fluid.interface_coordinates = interface_coordinates(interface)
solid.interface_coordinates = interface_coordinates(interface)
interface.interface_coordinates = interface_coordinates(interface)
for coordinate_index in range(4):
    fluid.interface_table.append(
        interface_coordinates_transfer_table(
            fluid, fluid.interface_coordinates, coordinate_index
        )
    )
    solid.interface_table.append(
        interface_coordinates_transfer_table(
            solid, solid.interface_coordinates, coordinate_index
        )
    )
    fluid.interface_table_collapse.append(
        interface_coordinates_transfer_table(
            fluid, fluid.interface_coordinates, coordinate_index, True
        )
    )
    solid.interface_table_collapse.append(
        interface_coordinates_transfer_table(
            solid, solid.interface_coordinates, coordinate_index, True
        )
    )
fluid.interface_table.append(
    interface_coordinates_transfer_table(fluid, fluid.interface_coordinates, 4)
)
interface.interface_table = extension_interface_coordinates_transfer_table(
    interface, interface.interface_coordinates
)

# Define variational forms
fluid.primal_problem = Problem(bilinear_form_fluid, functional_fluid)
fluid.adjoint_problem = Problem(
    bilinear_form_fluid_adjoint,
    functional_fluid_adjoint,
)
solid.primal_problem = Problem(
    bilinear_form_solid,
    functional_solid,
    functional_solid_interface,
)
solid.adjoint_problem = Problem(
    bilinear_form_solid_adjoint,
    functional_solid_adjoint,
)

# Create time interval structures
fluid_timeline = TimeLine()
fluid_timeline.unify(
    param.TIME_STEP,
    param.LOCAL_MESH_SIZE_FLUID,
    param.GLOBAL_MESH_SIZE,
    param.INITIAL_TIME,
)
solid_timeline = TimeLine()
solid_timeline.unify(
    param.TIME_STEP,
    param.LOCAL_MESH_SIZE_SOLID,
    param.GLOBAL_MESH_SIZE,
    param.INITIAL_TIME,
)

# Set deoupling method
if param.RELAXATION:

    decoupling = relaxation

else:

    decoupling = shooting

# Refine time meshes
fluid_size = fluid_timeline.size_global - fluid_timeline.size
solid_size = solid_timeline.size_global - solid_timeline.size
for i in range(param.REFINEMENT_LEVELS):

    fluid_refinements_txt = open(
        f"fluid_{fluid_size}-{solid_size}_refinements.txt", "r"
    )
    solid_refinements_txt = open(
        f"solid_{fluid_size}-{solid_size}_refinements.txt", "r"
    )
    fluid_refinements = [bool(int(x)) for x in fluid_refinements_txt.read()]
    solid_refinements = [bool(int(x)) for x in solid_refinements_txt.read()]
    fluid_refinements_txt.close()
    solid_refinements_txt.close()
    fluid_timeline.refine(fluid_refinements)
    solid_timeline.refine(solid_refinements)
    split(fluid_timeline, solid_timeline)
    fluid_size = fluid_timeline.size_global - fluid_timeline.size
    solid_size = solid_timeline.size_global - solid_timeline.size
    print(f"Global number of macro time-steps: {fluid_timeline.size}")
    print(
        f"Global number of micro time-steps in the fluid timeline: {fluid_size}"
    )
    print(
        f"Global number of micro time-steps in the solid timeline: {solid_size}"
    )

# fluid_timeline.print()
solid_timeline.print()

# Create directory
fluid_size = fluid_timeline.size_global - fluid_timeline.size
solid_size = solid_timeline.size_global - solid_timeline.size
try:

    os.makedirs(f"{fluid_size}-{solid_size}")

except FileExistsError:

    pass
os.chdir(f"{fluid_size}-{solid_size}")

# Perform time-stepping of the primal problem
adjoint = False
start = time.time()
if param.COMPUTE_PRIMAL:
    time_stepping(
        fluid,
        solid,
        interface,
        param,
        decoupling,
        fluid_timeline,
        solid_timeline,
        adjoint,
    )
end = time.time()
print(end - start)
fluid_timeline.load(fluid, "fluid", adjoint)
solid_timeline.load(solid, "solid", adjoint)

goal_functional = goal_functional_solid(solid, solid_timeline, param)
print(f"Value of goal functional: {sum(goal_functional)}")

# Perform time-stepping of the adjoint problem
adjoint = True
if param.COMPUTE_ADJOINT:
    time_stepping(
        fluid,
        solid,
        interface,
        param,
        decoupling,
        fluid_timeline,
        solid_timeline,
        adjoint,
    )
fluid_timeline.load(fluid, "fluid", adjoint)
solid_timeline.load(solid, "solid", adjoint)

# Compute residuals
(
    primal_residual_fluid,
    primal_residual_solid,
    adjoint_residual_fluid,
    adjoint_residual_solid,
    goal_functional,
) = compute_residuals(fluid, solid, param, fluid_timeline, solid_timeline)

# Refine mesh
refine_time(
    primal_residual_fluid,
    primal_residual_solid,
    adjoint_residual_fluid,
    adjoint_residual_solid,
    fluid_timeline,
    solid_timeline,
)
