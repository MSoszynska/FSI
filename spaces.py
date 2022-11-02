from fenics import (
    near,
    SubDomain,
    CellDiameter,
    FacetNormal,
    Measure,
    MeshFunction,
    FiniteElement,
    VectorElement,
    FunctionSpace,
    Constant,
    Expression,
    DirichletBC,
    between,
    MixedElement,
)

# Define boundary parts
def noslip_boundary(x, on_boundary):
    return near(x[1], 0.0, 1.0e-4) or near(x[1], 0.41, 1.0e-4)


def inflow_boundary(x, on_boundary):
    return near(x[0], 0.0)


def cylinder_boundary(x, on_boundary):
    return (
        on_boundary
        and between(x[0], (0.15, 0.25))
        and between(x[1], (0.15, 0.25))
    )


def outflow_boundary(x, on_boundary):
    return near(x[0], 2.5, 1.0e-4)


def obstacle(x, on_boundary):
    return (
        on_boundary
        and between(x[0], (0.15, 0.65))
        and between(x[1], (0.18, 0.22))
    )


def boundary(x, on_boundary):
    return on_boundary


# Define interface
class Inner_boundary(SubDomain):
    def inside(self, x, on_boundary):

        return near(x[1], 0.19) or near(x[1], 0.21) or near(x[0], 0.6)


# Define outflow
class Outflow(SubDomain):
    def inside(self, x, on_boundary):

        return near(x[0], 2.5)


# Define Dirichlet boundary conditions
def zero_constant(dim):
    if dim == 1:
        return Constant(0.0)
    elif dim == 2:
        return Constant((0.0, 0.0))
    elif dim == 3:
        return Constant((0.0, 0.0, 0.0))


# Store space attributes
class Space:
    def __init__(self, mesh, dimension, degree, name):

        # Define mesh parameters
        self.mesh = mesh
        self.cell_size = CellDiameter(mesh)
        self.normal_vector = FacetNormal(mesh)

        # Define measures
        inner_boundary = Inner_boundary()
        outflow = Outflow()
        sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        sub_domains.set_all(0)
        inner_boundary.mark(sub_domains, 1)
        outflow.mark(sub_domains, 2)
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=sub_domains)

        # Define function spaces
        finite_element = []
        if name == "fluid":
            number_of_variables = 3
        else:
            number_of_variables = 2
        for i in range(number_of_variables):
            if dimension[i] > 1:
                finite_element.append(
                    VectorElement("CG", mesh.ufl_cell(), degree[i])
                )
            else:
                finite_element.append(
                    FiniteElement("CG", mesh.ufl_cell(), degree[i])
                )
        if name == "fluid":
            self.function_space = FunctionSpace(
                mesh,
                MixedElement(
                    [finite_element[0], finite_element[1], finite_element[2]]
                ),
            )
            self.function_space_split = [
                self.function_space.sub(0).collapse(),
                self.function_space.sub(1).collapse(),
                self.function_space.sub(2).collapse(),
            ]
            self.function_space_correction = FunctionSpace(
                mesh,
                MixedElement([finite_element[0], finite_element[2]]),
            )
        else:
            self.function_space = FunctionSpace(
                mesh, finite_element[0] * finite_element[1]
            )
            self.function_space_split = [
                self.function_space.sub(0).collapse(),
                self.function_space.sub(1).collapse(),
            ]
        self.function_space_split_one_dimension = [
            FunctionSpace(mesh, "CG", 2),
            FunctionSpace(mesh, "CG", 1),
        ]

        # Initialize coordinates tables
        self.interface_coordinates = None
        self.interface_table = []
        self.interface_table_collapse = []

        # Define additional information
        self.dimension = dimension
        self.degree = degree
        self.name = name

    # Define boundaries
    def boundaries(self, time, param, adjoint, correction_space=False):

        if not adjoint:
            if time < 2.0:
                inflow_function = Expression(
                    (
                        "3.0 / 0.1681 * (1.0 - cos(pi / 2.0 * time)) * x[1] * (0.41 - x[1]) * mean_velocity",
                        "0.0",
                    ),
                    time=time,
                    mean_velocity=param.MEAN_VELOCITY,
                    degree=2,
                )
            else:
                inflow_function = Expression(
                    (
                        "6.0 / 0.1681 * x[1] * (0.41 - x[1]) * mean_velocity",
                        "0.0",
                    ),
                    time=time,
                    mean_velocity=param.MEAN_VELOCITY,
                    degree=2,
                )
        else:
            inflow_function = Constant((0.0, 0.0))
        if self.name == "fluid" and not correction_space:

            return [
                DirichletBC(
                    self.function_space.sub(0),
                    zero_constant(self.dimension[0]),
                    noslip_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(0),
                    zero_constant(self.dimension[0]),
                    cylinder_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(0),
                    inflow_function,
                    inflow_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(1),
                    zero_constant(self.dimension[0]),
                    noslip_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(1),
                    zero_constant(self.dimension[0]),
                    cylinder_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(1),
                    zero_constant(self.dimension[0]),
                    inflow_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(1),
                    zero_constant(self.dimension[0]),
                    outflow_boundary,
                ),
            ]
        elif self.name == "fluid":

            return [
                DirichletBC(
                    self.function_space_correction.sub(0),
                    zero_constant(self.dimension[0]),
                    noslip_boundary,
                ),
                DirichletBC(
                    self.function_space_correction.sub(0),
                    zero_constant(self.dimension[0]),
                    cylinder_boundary,
                ),
                DirichletBC(
                    self.function_space_correction.sub(0),
                    inflow_function,
                    inflow_boundary,
                ),
            ]
        else:

            return [
                DirichletBC(
                    self.function_space.sub(0),
                    zero_constant(self.dimension[0]),
                    cylinder_boundary,
                ),
                DirichletBC(
                    self.function_space.sub(1),
                    zero_constant(self.dimension[1]),
                    cylinder_boundary,
                ),
            ]
