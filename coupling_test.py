import unittest as ut
from fenics import (Constant, Point, project, assemble, dot, BoundaryMesh, SubDomain, SubMesh, near, File, Expression,
                    MeshFunction, cells, refine, VectorFunctionSpace, ALE)
from mshr import Polygon, generate_mesh, Circle, Rectangle
from spaces import Space
from coupling import (
    interface_coordinates,
    interface_coordinates_transfer_table,
    extension_interface_coordinates_transfer_table,
    mirror_function
)

class TestMirror(ut.TestCase):

    def test_mirror(self):

        # Define meshes
        domain_fluid = Polygon(
            [
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(1.0, 1.0),
                Point(0.0, 1.0),
            ]
        )
        domain_solid = Polygon(
            [
                Point(0.0, 0.0),
                Point(0.0, -1.0),
                Point(1.0, -1.0),
                Point(1.0, 0.0),
            ]
        )
        mesh_fluid = generate_mesh(domain_fluid, 10)
        mesh_solid = generate_mesh(domain_solid, 10)

        # Define interface
        class Inner_boundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], 0.0)

        # Define interface mesh
        boundary_mesh = BoundaryMesh(mesh_fluid, "exterior")
        inner_boundary = Inner_boundary()
        mesh_interface = SubMesh(boundary_mesh, inner_boundary)

        # Create function spaces
        fluid_dimension = [2, 2, 1]
        fluid_degree = [2, 2, 1]
        solid_dimension = [2, 2]
        solid_degree = [2, 2]
        fluid = Space(mesh_fluid, fluid_dimension, fluid_degree, "fluid", inner_boundary)
        solid = Space(mesh_solid, solid_dimension, solid_degree, "solid", inner_boundary)
        interface = Space(mesh_interface, solid_dimension, solid_degree, "interface", inner_boundary)

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

        function_test = Expression(("1.0", "0.0"), degree=0)
        function_fluid = project(function_test, fluid.function_space_split[0])
        function_solid = project(function_test, solid.function_space_split[0])
        function_fluid_to_solid = mirror_function(function_fluid, solid, fluid)
        function_solid_to_fluid = mirror_function(function_solid, fluid, solid)
        result_fluid = assemble(dot(function_fluid - function_solid_to_fluid, function_fluid) * fluid.ds(1))
        result_solid = assemble(dot(function_solid - function_fluid_to_solid, function_solid) * solid.ds(1))
        self.assertAlmostEqual(result_fluid, 0.0)
        self.assertAlmostEqual(result_solid, 0.0)


