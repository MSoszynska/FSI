from fenics import (
    dot,
    grad,
    Expression,
    project,
    Constant,
    inner,
    div,
    Identity,
    interpolate,
    Function,
    assign,
    assemble,
    derivative,
    det,
    tr,
    inv,
)
from math import sqrt
from parameters import Parameters
from spaces import Space
from time_structure import MicroTimeStep
from coupling import mirror_function

# Define variational problem
class Problem:
    def __init__(
        self,
        bilinear_form,
        functional,
        functional_interface=None,
    ):
        self.bilinear_form = bilinear_form
        self.functional = functional
        self.functional_interface = functional_interface


# Define deformation gradient
def deformation_gradient(displacement):
    dimension = displacement.geometric_dimension()
    return Identity(dimension) + grad(displacement)


# Define strain
def epsilon(displacement):
    dimension = displacement.geometric_dimension()
    return 0.5 * (
        deformation_gradient(displacement).T
        * deformation_gradient(displacement)
        - Identity(dimension)
    )


# Define fluid stress
def sigma_fluid(velocity, displacement, pressure, fluid, param, scheme=None):
    if scheme is None:
        explicit = True
        implicit = True
    else:
        explicit = scheme[0]
        implicit = scheme[1]
    dimension = fluid.dimension[0]
    return explicit * param.RHO_FLUID * param.NU * (
        grad(velocity) * inv(deformation_gradient(displacement))
        + inv(deformation_gradient(displacement)).T * grad(velocity).T
    ) - implicit * pressure * Identity(dimension)


# Define solid stress
def sigma_solid(displacement, solid, param):
    dimension = solid.dimension[0]
    return deformation_gradient(displacement) * (
        param.ZETA * tr(epsilon(displacement)) * Identity(dimension)
        + 2.0 * param.MU * epsilon(displacement)
    )


# Define characteristic functions corresponding to chosen functionals
def characteristic_function_fluid(param: Parameters):

    return Expression(
        "functional", functional=param.GOAL_FUNCTIONAL_FLUID, degree=0
    )


def characteristic_function_solid(param: Parameters):

    return Expression(
        "functional", functional=param.GOAL_FUNCTIONAL_SOLID, degree=0
    )


def external_force(time, param):

    height = 0.0
    threshold = 0.1
    width = 0.02
    function = Expression(("0.0", "-height"), height=height, degree=0)
    # if int(time) + threshold <= time and time <= int(time) + threshold + width:
    #     function = Expression(
    #         ("0.0", "-height * (threshold + width - time) / width"),
    #         height=height,
    #         threshold=threshold,
    #         width=width,
    #         time=time,
    #         degree=1,
    #     )
    if int(time) + threshold + width < time:
        function = Expression(("0.0", "0.0"), degree=0)

    return function


# Define coefficients of 2-point Gaussian quadrature
def gauss(microtimestep, microtimestep_before=None):

    t_new = microtimestep.point
    if microtimestep_before is None:
        t_old = microtimestep.before.point
        dt = microtimestep.before.dt
    else:
        t_old = microtimestep_before.point
        dt = microtimestep.point - microtimestep_before.point
    t_average = 0.5 * (t_old + t_new)
    return [
        dt / (2.0 * sqrt(3)) + t_average,
        -dt / (2.0 * sqrt(3)) + t_average,
    ]


# Define goal functional
def goal_functional(
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    function_name,
    space,
    initial=False,
):

    if initial:
        zero_function = Function(space.function_space_split[0])
        return [zero_function, zero_function]
    else:
        function_before = microtimestep_before.functions[function_name]
        time_before = microtimestep_before.point
        function = microtimestep.functions[function_name]
        time = microtimestep.point

        def linear_extrapolation(time_gauss):
            return (function - function_before) / (
                time - time_before
            ) * time_gauss + (
                function_before * time - function * time_before
            ) / (
                time - time_before
            )

        time_gauss_1, time_gauss_2 = gauss(microtimestep, microtimestep_before)

        return [
            0.5 * linear_extrapolation(time_gauss_1) * (-time_gauss_1 + time)
            + 0.5
            * linear_extrapolation(time_gauss_2)
            * (-time_gauss_2 + time),
            0.5
            * linear_extrapolation(time_gauss_1)
            * (time_gauss_1 - time_before)
            + 0.5
            * linear_extrapolation(time_gauss_2)
            * (time_gauss_2 - time_before),
        ]


def create_adjoint(
    primal_functions,
    adjoint_functions,
    test_functions,
    form,
    space: Space,
    param: Parameters,
    interface_problem,
    functional,
    scheme=None,
    error_estimate=False,
):

    if error_estimate:
        velocity_primal = project(
            primal_functions[0], space.function_space_split[0]
        )
        displacement_primal = project(
            primal_functions[1], space.function_space_split[1]
        )
        first_test_component = project(
            test_functions[0], space.function_space_split[0]
        )
        second_test_component = project(
            test_functions[1], space.function_space_split[1]
        )
        if not functional:
            velocity_adjoint = project(
                adjoint_functions[0], space.function_space_split[0]
            )
            displacement_adjoint = project(
                adjoint_functions[1], space.function_space_split[1]
            )
        if space.name == "fluid":
            if not functional:
                pressure_primal = project(
                    primal_functions[2], space.function_space_split[2]
                )
                third_test_component = project(
                    test_functions[2], space.function_space_split[2]
                )
            if not interface_problem:
                if not functional:
                    pressure_adjoint = project(
                        adjoint_functions[2], space.function_space_split[2]
                    )
    else:
        velocity_primal = primal_functions[0]
        displacement_primal = primal_functions[1]
        first_test_component = test_functions[0]
        second_test_component = test_functions[1]
        if not functional:
            velocity_adjoint = adjoint_functions[0]
            displacement_adjoint = adjoint_functions[1]
        if space.name == "fluid":
            if not functional:
                pressure_primal = primal_functions[2]
                third_test_component = test_functions[2]
            if not interface_problem:
                if not functional:
                    pressure_adjoint = adjoint_functions[2]
    if space.name == "fluid":
        if functional:
            form_primal = form(velocity_primal, displacement_primal, param)
        elif interface_problem:
            form_primal = form(
                velocity_primal,
                displacement_primal,
                pressure_primal,
                velocity_adjoint,
                displacement_adjoint,
                space,
                param,
                scheme,
            )
        else:
            form_primal = form(
                velocity_primal,
                displacement_primal,
                pressure_primal,
                velocity_adjoint,
                displacement_adjoint,
                pressure_adjoint,
                space,
                param,
                scheme,
            )
    if space.name == "solid":
        form_primal = form(
            velocity_primal,
            displacement_primal,
            velocity_adjoint,
            displacement_adjoint,
            space,
            param,
        )
    if space.name == "fluid":
        if functional:
            form_adjoint = derivative(
                form_primal,
                (velocity_primal, displacement_primal),
                (first_test_component, second_test_component),
            )
        else:
            form_adjoint = derivative(
                form_primal,
                (velocity_primal, displacement_primal, pressure_primal),
                (
                    first_test_component,
                    second_test_component,
                    third_test_component,
                ),
            )
    if space.name == "solid":
        form_adjoint = derivative(
            form_primal,
            (velocity_primal, displacement_primal),
            (first_test_component, second_test_component),
        )
    return form_adjoint


# Define variational forms of the fluid subproblem
def form_fluid(
    velocity_primal_fluid,
    displacement_primal_fluid,
    pressure_primal_fluid,
    first_test_function,
    second_test_function,
    third_test_function,
    fluid: Space,
    param: Parameters,
    scheme=None,
):
    if scheme is None:
        explicit = True
        implicit = True
    else:
        explicit = scheme[0]
        implicit = scheme[1]

    extension = Expression(
        "1 +20*exp(-15*(x[0]-0.6)*(x[0]-0.6))"
        "*exp(-200*(x[1]-0.21)*(x[1]-0.21))+20"
        "*exp(-15*(x[0]-0.6)*(x[0]-0.6))"
        "*exp(-200*(x[1]-0.19)*(x[1]-0.19))",
        degree=2,
    )

    # extension = Expression(
    #     "1 + 100*exp(-75*(x[0]-0.6)*(x[0]-0.6))"
    #     "*exp(-1000*(x[1]-0.21)*(x[1]-0.21))+ 100"
    #     "*exp(-75*(x[0]-0.6)*(x[0]-0.6))"
    #     "*exp(-1000*(x[1]-0.19)*(x[1]-0.19))",
    #     degree=2,
    # )
    return (
        det(deformation_gradient(displacement_primal_fluid))
        * inner(
            sigma_fluid(
                velocity_primal_fluid,
                displacement_primal_fluid,
                pressure_primal_fluid,
                fluid,
                param,
                scheme,
            )
            * inv(deformation_gradient(displacement_primal_fluid)).T,
            grad(first_test_function),
        )
        * fluid.dx
        + explicit
        * param.RHO_FLUID
        * det(deformation_gradient(displacement_primal_fluid))
        * dot(
            dot(
                grad(velocity_primal_fluid)
                * inv(deformation_gradient(displacement_primal_fluid)),
                velocity_primal_fluid,
            ),
            first_test_function,
        )
        * fluid.dx
        + implicit
        * det(deformation_gradient(displacement_primal_fluid))
        * inner(
            inv(deformation_gradient(displacement_primal_fluid)),
            grad(velocity_primal_fluid).T,
        )
        * third_test_function
        * fluid.dx
        + explicit
        * extension
        * inner(grad(displacement_primal_fluid), grad(second_test_function))
        * fluid.dx
        - det(deformation_gradient(displacement_primal_fluid))
        * dot(
            dot(
                sigma_fluid(
                    velocity_primal_fluid,
                    displacement_primal_fluid,
                    pressure_primal_fluid,
                    fluid,
                    param,
                    scheme,
                )
                * inv(deformation_gradient(displacement_primal_fluid)).T,
                fluid.normal_vector,
            ),
            first_test_function,
        )
        * fluid.ds(1)
        + explicit
        * param.RHO_FLUID
        * param.NU
        * param.GAMMA
        / fluid.cell_size
        * dot(velocity_primal_fluid, first_test_function)
        * fluid.ds(1)
        + explicit
        * param.GAMMA
        / fluid.cell_size
        * dot(displacement_primal_fluid, second_test_function)
        * fluid.ds(1)
    )


def form_fluid_interface(
    velocity_primal_interface,
    displacement_primal_interface,
    first_test_function,
    second_test_function,
    space,
    param,
):
    return param.RHO_FLUID * param.NU * param.GAMMA / space.cell_size * dot(
        velocity_primal_interface, first_test_function
    ) * space.ds(1) + param.GAMMA / space.cell_size * dot(
        displacement_primal_interface, second_test_function
    ) * space.ds(
        1
    )


def form_fluid_velocity_derivative(
    velocity_fluid,
    displacement_fluid,
    param: Parameters,
):

    return param.RHO_FLUID * det(deformation_gradient(displacement_fluid))


def form_fluid_displacement_derivative(
    velocity_fluid,
    displacement_fluid,
    param: Parameters,
):
    dimension = displacement_fluid.geometric_dimension()
    return (
        param.RHO_FLUID
        * det(deformation_gradient(displacement_fluid))
        * grad(velocity_fluid)
        * inv(deformation_gradient(displacement_fluid))
    )


def bilinear_form_fluid(
    velocity_primal_fluid,
    displacement_primal_fluid,
    pressure_primal_fluid,
    first_test_function,
    second_test_function,
    third_test_function,
    velocity_primal_fluid_before,
    displacement_primal_fluid_before,
    pressure_primal_fluid_before,
    fluid: Space,
    param: Parameters,
    time_step_size,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
):

    theta = param.THETA + time_step_size

    return (
        (
            form_fluid_velocity_derivative(
                velocity_primal_fluid, displacement_primal_fluid, param
            )
        )
        * dot(velocity_primal_fluid, first_test_function)
        * fluid.dx
        - form_fluid_velocity_derivative(
            velocity_primal_fluid, displacement_primal_fluid, param
        )
        * dot(velocity_primal_fluid_before, first_test_function)
        * fluid.dx
        - dot(
            dot(
                (
                    theta
                    * form_fluid_displacement_derivative(
                        velocity_primal_fluid, displacement_primal_fluid, param
                    )
                    + (1.0 - theta)
                    * form_fluid_displacement_derivative(
                        velocity_primal_fluid_before,
                        displacement_primal_fluid_before,
                        param,
                    )
                ),
                displacement_primal_fluid,
            ),
            first_test_function,
        )
        * fluid.dx
        + theta
        * dot(
            dot(
                form_fluid_displacement_derivative(
                    velocity_primal_fluid, displacement_primal_fluid, param
                ),
                displacement_primal_fluid_before,
            ),
            first_test_function,
        )
        * fluid.dx
        + theta
        * time_step_size
        * form_fluid(
            velocity_primal_fluid,
            displacement_primal_fluid,
            pressure_primal_fluid,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[True, False],
        )
        + time_step_size
        * form_fluid(
            velocity_primal_fluid,
            displacement_primal_fluid,
            pressure_primal_fluid,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[False, True],
        )
    )


def functional_fluid(
    velocity_primal_fluid_before,
    displacement_primal_fluid_before,
    pressure_primal_fluid_before,
    velocity_primal_solid,
    displacement_primal_solid,
    velocity_primal_solid_before,
    displacement_primal_solid_before,
    first_test_function,
    second_test_function,
    third_test_function,
    fluid: Space,
    solid: Space,
    param: Parameters,
    time,
    time_before,
    time_step_size,
    time_step_size_before,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
    initial,
):

    theta = param.THETA + time_step_size

    velocity_primal_interface = mirror_function(
        velocity_primal_solid, fluid, solid
    )
    velocity_primal_interface_before = mirror_function(
        velocity_primal_solid_before, fluid, solid
    )
    displacement_primal_interface = mirror_function(
        displacement_primal_solid, fluid, solid
    )
    displacement_primal_interface_before = mirror_function(
        displacement_primal_solid_before, fluid, solid
    )

    return (
        -(1.0 - theta)
        * dot(
            dot(
                form_fluid_displacement_derivative(
                    velocity_primal_fluid_before,
                    displacement_primal_fluid_before,
                    param,
                ),
                displacement_primal_fluid_before,
            ),
            first_test_function,
        )
        * fluid.dx
        - (1.0 - theta)
        * time_step_size
        * form_fluid(
            velocity_primal_fluid_before,
            displacement_primal_fluid_before,
            pressure_primal_fluid_before,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[True, False],
        )
        + theta
        * time_step_size
        * form_fluid_interface(
            velocity_primal_interface,
            displacement_primal_interface,
            first_test_function,
            second_test_function,
            fluid,
            param,
        )
        + (1.0 - theta)
        * time_step_size
        * form_fluid_interface(
            velocity_primal_interface_before,
            displacement_primal_interface_before,
            first_test_function,
            second_test_function,
            fluid,
            param,
        )
    )


def bilinear_form_fluid_correction(
    velocity_primal_fluid_correction,
    pressure_primal_fluid_correction,
    velocity_primal_fluid_between,
    displacement_primal_fluid_between,
    pressure_primal_fluid_between,
    first_test_function,
    second_test_function,
    fluid: Space,
    param: Parameters,
):
    dimension = displacement_primal_fluid_between.geometric_dimension()
    return (
        inner(
            det(deformation_gradient(displacement_primal_fluid_between))
            * (
                grad(velocity_primal_fluid_correction)
                * inv(deformation_gradient(displacement_primal_fluid_between))
                - pressure_primal_fluid_correction * Identity(dimension)
            )
            * inv(deformation_gradient(displacement_primal_fluid_between)).T,
            grad(first_test_function),
        )
        * fluid.dx
        + det(deformation_gradient(displacement_primal_fluid_between))
        * inner(
            inv(deformation_gradient(displacement_primal_fluid_between)),
            grad(velocity_primal_fluid_correction).T,
        )
        * second_test_function
        * fluid.dx
    )


def functional_fluid_correction(
    velocity_primal_fluid_between,
    displacement_primal_fluid_between,
    pressure_primal_fluid_between,
    first_test_function,
    second_test_function,
    fluid: Space,
    param: Parameters,
):
    dimension = displacement_primal_fluid_between.geometric_dimension()
    return (
        inner(
            det(deformation_gradient(displacement_primal_fluid_between))
            * (
                grad(velocity_primal_fluid_between)
                * inv(deformation_gradient(displacement_primal_fluid_between))
                - pressure_primal_fluid_between * Identity(dimension)
            )
            * inv(deformation_gradient(displacement_primal_fluid_between)).T,
            grad(first_test_function),
        )
        * fluid.dx
    )


def form_fluid_adjoint(
    velocity_fluid_primal,
    displacement_fluid_primal,
    pressure_fluid_primal,
    velocity_fluid_adjoint,
    displacement_fluid_adjoint,
    pressure_fluid_adjoint,
    first_test_function,
    second_test_function,
    third_test_function,
    fluid: Space,
    param: Parameters,
    scheme=None,
    error_estimate=False,
):
    primal_functions = [
        velocity_fluid_primal,
        displacement_fluid_primal,
        pressure_fluid_primal,
    ]
    adjoint_functions = [
        velocity_fluid_adjoint,
        displacement_fluid_adjoint,
        pressure_fluid_adjoint,
    ]
    test_functions = [
        first_test_function,
        second_test_function,
        third_test_function,
    ]
    form_adjoint = create_adjoint(
        primal_functions,
        adjoint_functions,
        test_functions,
        form_fluid,
        fluid,
        param,
        False,
        False,
        scheme=scheme,
        error_estimate=error_estimate,
    )
    return form_adjoint


def form_fluid_interface_adjoint(
    velocity_fluid_primal,
    displacement_fluid_primal,
    pressure_fluid_primal,
    velocity_fluid_adjoint,
    displacement_fluid_adjoint,
    first_test_function,
    second_test_function,
    third_test_function,
    fluid: Space,
    param: Parameters,
    scheme=None,
    error_estimate=False,
):
    primal_functions = [
        velocity_fluid_primal,
        displacement_fluid_primal,
        pressure_fluid_primal,
    ]
    adjoint_functions = [velocity_fluid_adjoint, displacement_fluid_adjoint]
    test_functions = [
        first_test_function,
        second_test_function,
        third_test_function,
    ]
    form_adjoint = create_adjoint(
        primal_functions,
        adjoint_functions,
        test_functions,
        form_solid_interface,
        fluid,
        param,
        True,
        False,
        scheme=scheme,
        error_estimate=error_estimate,
    )
    return form_adjoint


def form_fluid_velocity_derivative_adjoint(
    velocity_fluid_primal,
    displacement_fluid_primal,
    first_test_function,
    second_test_function,
    fluid: Space,
    param: Parameters,
    error_estimate=False,
):
    primal_functions = [velocity_fluid_primal, displacement_fluid_primal]
    adjoint_functions = None
    test_functions = [first_test_function, second_test_function]
    form_adjoint = create_adjoint(
        primal_functions,
        adjoint_functions,
        test_functions,
        form_fluid_velocity_derivative,
        fluid,
        param,
        False,
        True,
        scheme=None,
        error_estimate=error_estimate,
    )
    return form_adjoint


def form_fluid_displacement_derivative_adjoint(
    velocity_fluid_primal,
    displacement_fluid_primal,
    first_test_function,
    second_test_function,
    fluid: Space,
    param: Parameters,
    error_estimate=False,
):
    primal_functions = [velocity_fluid_primal, displacement_fluid_primal]
    adjoint_functions = None
    test_functions = [first_test_function, second_test_function]
    form_adjoint = create_adjoint(
        primal_functions,
        adjoint_functions,
        test_functions,
        form_fluid_displacement_derivative,
        fluid,
        param,
        False,
        True,
        scheme=None,
        error_estimate=error_estimate,
    )
    return form_adjoint


def bilinear_form_fluid_adjoint(
    velocity_adjoint_fluid,
    displacement_adjoint_fluid,
    pressure_adjoint_fluid,
    first_test_function,
    second_test_function,
    third_test_function,
    velocity_adjoint_fluid_after,
    displacement_adjoint_fluid_after,
    pressure_adjoint_fluid_after,
    fluid: Space,
    param: Parameters,
    time_step_size,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    error_estimate=False,
):

    theta = param.THETA + time_step_size

    primal_solution = Function(fluid.function_space)
    assign(primal_solution.sub(0), microtimestep.functions["primal_velocity"])
    assign(
        primal_solution.sub(1), microtimestep.functions["primal_displacement"]
    )
    assign(primal_solution.sub(2), microtimestep.functions["primal_pressure"])
    primal_solution_before = Function(fluid.function_space)
    assign(
        primal_solution_before.sub(0),
        microtimestep_before.functions["primal_velocity"],
    )
    assign(
        primal_solution_before.sub(1),
        microtimestep_before.functions["primal_displacement"],
    )
    assign(
        primal_solution_before.sub(2),
        microtimestep_before.functions["primal_pressure"],
    )
    for boundary in fluid.boundaries(0.0, param, True):
        boundary.apply(primal_solution.vector())
        boundary.apply(primal_solution_before.vector())
    (
        velocity_fluid_primal,
        displacement_fluid_primal,
        pressure_fluid_primal,
    ) = primal_solution.split()
    (
        velocity_fluid_primal_before,
        displacement_fluid_primal_before,
        pressure_fluid_primal_before,
    ) = primal_solution_before.split()

    return (
        +form_fluid_velocity_derivative_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            first_test_function,
            second_test_function,
            fluid,
            param,
            error_estimate=error_estimate,
        )
        * dot(
            velocity_fluid_primal - velocity_fluid_primal_before,
            velocity_adjoint_fluid,
        )
        * fluid.dx
        + (
            form_fluid_velocity_derivative(
                velocity_fluid_primal, displacement_fluid_primal, param
            )
        )
        * dot(first_test_function, velocity_adjoint_fluid)
        * fluid.dx
        - dot(
            dot(
                theta
                * form_fluid_displacement_derivative_adjoint(
                    velocity_fluid_primal,
                    displacement_fluid_primal,
                    first_test_function,
                    second_test_function,
                    fluid,
                    param,
                    error_estimate=error_estimate,
                ),
                displacement_fluid_primal - displacement_fluid_primal_before,
            ),
            velocity_adjoint_fluid,
        )
        * fluid.dx
        - dot(
            dot(
                (
                    theta
                    * form_fluid_displacement_derivative(
                        velocity_fluid_primal, displacement_fluid_primal, param
                    )
                    + (1.0 - theta)
                    * form_fluid_displacement_derivative(
                        velocity_fluid_primal_before,
                        displacement_fluid_primal_before,
                        param,
                    )
                ),
                second_test_function,
            ),
            velocity_adjoint_fluid,
        )
        * fluid.dx
        + theta
        * time_step_size
        * form_fluid_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            pressure_fluid_primal,
            velocity_adjoint_fluid,
            displacement_adjoint_fluid,
            pressure_adjoint_fluid,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[True, False],
            error_estimate=error_estimate,
        )
        + time_step_size
        * form_fluid_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            pressure_fluid_primal,
            velocity_adjoint_fluid,
            displacement_adjoint_fluid,
            pressure_adjoint_fluid,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[False, True],
            error_estimate=error_estimate,
        )
    )


def functional_fluid_adjoint(
    velocity_adjoint_fluid_after,
    displacement_adjoint_fluid_after,
    pressure_adjoint_fluid_after,
    velocity_adjoint_solid,
    displacement_adjoint_solid,
    velocity_adjoint_solid_after,
    displacement_adjoint_solid_after,
    first_test_function,
    second_test_function,
    third_test_function,
    fluid: Space,
    solid: Space,
    param: Parameters,
    time,
    time_after,
    time_step_size,
    time_step_size_after,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
    initial,
    error_estimate=False,
):

    theta = param.THETA + time_step_size
    theta_after = param.THETA + time_step_size_after

    if initial:
        velocity_adjoint_fluid_after = Function(fluid.function_space_split[0])
        displacement_adjoint_fluid_after = Function(
            fluid.function_space_split[1]
        )
        pressure_adjoint_fluid_after = Function(fluid.function_space_split[2])
        velocity_adjoint_solid_after = Function(solid.function_space_split[0])
        displacement_adjoint_solid_after = Function(
            solid.function_space_split[1]
        )
    primal_solution = Function(fluid.function_space)
    assign(primal_solution.sub(0), microtimestep.functions["primal_velocity"])
    assign(
        primal_solution.sub(1), microtimestep.functions["primal_displacement"]
    )
    assign(primal_solution.sub(2), microtimestep.functions["primal_pressure"])
    primal_solution_after = Function(fluid.function_space)
    assign(
        primal_solution_after.sub(0),
        microtimestep_after.functions["primal_velocity"],
    )
    assign(
        primal_solution_after.sub(1),
        microtimestep_after.functions["primal_displacement"],
    )
    assign(
        primal_solution_after.sub(2),
        microtimestep_after.functions["primal_pressure"],
    )
    for boundary in fluid.boundaries(0.0, param, True):
        boundary.apply(primal_solution.vector())
        boundary.apply(primal_solution_after.vector())
    (
        velocity_fluid_primal,
        displacement_fluid_primal,
        pressure_fluid_primal,
    ) = primal_solution.split()
    (
        velocity_fluid_primal_after,
        displacement_fluid_primal_after,
        pressure_fluid_primal_after,
    ) = primal_solution_after.split()
    velocity_adjoint_interface = mirror_function(
        velocity_adjoint_solid, fluid, solid
    )
    velocity_adjoint_interface_after = mirror_function(
        velocity_adjoint_solid_after, fluid, solid
    )
    displacement_adjoint_interface = mirror_function(
        displacement_adjoint_solid, fluid, solid
    )
    displacement_adjoint_interface_after = mirror_function(
        displacement_adjoint_solid_after, fluid, solid
    )
    return (
        (
            form_fluid_velocity_derivative(
                velocity_fluid_primal_after,
                displacement_fluid_primal_after,
                param,
            )
        )
        * dot(first_test_function, velocity_adjoint_fluid_after)
        * fluid.dx
        + dot(
            dot(
                (1.0 - theta_after)
                * form_fluid_displacement_derivative_adjoint(
                    velocity_fluid_primal,
                    displacement_fluid_primal,
                    first_test_function,
                    second_test_function,
                    fluid,
                    param,
                    error_estimate=error_estimate,
                ),
                displacement_fluid_primal_after - displacement_fluid_primal,
            ),
            velocity_adjoint_fluid_after,
        )
        * fluid.dx
        - dot(
            dot(
                (
                    theta_after
                    * form_fluid_displacement_derivative(
                        velocity_fluid_primal_after,
                        displacement_fluid_primal_after,
                        param,
                    )
                    + (1.0 - theta_after)
                    * form_fluid_displacement_derivative(
                        velocity_fluid_primal, displacement_fluid_primal, param
                    )
                ),
                second_test_function,
            ),
            velocity_adjoint_fluid_after,
        )
        * fluid.dx
        - (1.0 - theta_after)
        * time_step_size_after
        * form_fluid_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            pressure_fluid_primal,
            velocity_adjoint_fluid_after,
            displacement_adjoint_fluid_after,
            pressure_adjoint_fluid_after,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[True, False],
            error_estimate=error_estimate,
        )
        - theta
        * time_step_size
        * form_fluid_interface_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            pressure_fluid_primal,
            velocity_adjoint_interface,
            displacement_adjoint_interface,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[True, False],
            error_estimate=error_estimate,
        )
        - (1.0 - theta_after)
        * time_step_size_after
        * form_fluid_interface_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            pressure_fluid_primal,
            velocity_adjoint_interface_after,
            displacement_adjoint_interface_after,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[True, False],
            error_estimate=error_estimate,
        )
        - time_step_size
        * form_fluid_interface_adjoint(
            velocity_fluid_primal,
            displacement_fluid_primal,
            pressure_fluid_primal,
            velocity_adjoint_interface,
            displacement_adjoint_interface,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            param,
            scheme=[False, True],
            error_estimate=error_estimate,
        )
        + 2.0
        * param.NU
        * characteristic_function_fluid(param)
        * dot(
            goal_functional(
                microtimestep_before,
                microtimestep,
                "primal_velocity",
                fluid,
            )[1],
            first_test_function,
        )
        * fluid.dx
        + 2.0
        * param.NU
        * characteristic_function_fluid(param)
        * dot(
            goal_functional(
                microtimestep,
                microtimestep_after,
                "primal_velocity",
                fluid,
                initial,
            )[0],
            first_test_function,
        )
        * fluid.dx
    )


# Define variational forms of the solid subproblem
def form_solid(
    velocity_primal_solid,
    displacement_primal_solid,
    first_test_function,
    second_test_function,
    solid: Space,
    param: Parameters,
):

    return (
        inner(
            sigma_solid(displacement_primal_solid, solid, param),
            grad(first_test_function),
        )
        * solid.dx
        - dot(velocity_primal_solid, second_test_function) * solid.dx
    )


def form_solid_interface(
    velocity_primal_interface,
    displacement_primal_interface,
    pressure_primal_interface,
    first_test_function,
    second_test_function,
    space,
    param,
    scheme=None,
):
    return (
        det(deformation_gradient(displacement_primal_interface))
        * dot(
            dot(
                sigma_fluid(
                    velocity_primal_interface,
                    displacement_primal_interface,
                    pressure_primal_interface,
                    space,
                    param,
                    scheme,
                )
                * inv(deformation_gradient(displacement_primal_interface)).T,
                space.normal_vector,
            ),
            first_test_function,
        )
        * space.ds(1)
    )


def bilinear_form_solid(
    velocity_primal_solid,
    displacement_primal_solid,
    first_test_function,
    second_test_function,
    solid: Space,
    param: Parameters,
    time_step_size,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
):

    theta = param.THETA + time_step_size

    return (
        param.RHO_SOLID
        * dot(velocity_primal_solid, first_test_function)
        * solid.dx
        + dot(displacement_primal_solid, second_test_function) * solid.dx
        + theta
        * time_step_size
        * form_solid(
            velocity_primal_solid,
            displacement_primal_solid,
            first_test_function,
            second_test_function,
            solid,
            param,
        )
    )


def functional_solid(
    velocity_primal_solid_before,
    displacement_primal_solid_before,
    velocity_primal_fluid,
    displacement_primal_fluid,
    pressure_primal_fluid,
    velocity_primal_fluid_before,
    displacement_primal_fluid_before,
    pressure_primal_fluid_before,
    first_test_function,
    second_test_function,
    solid: Space,
    fluid: Space,
    param: Parameters,
    time,
    time_before,
    time_step_size,
    time_step_size_before,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
    initial,
):

    theta = param.THETA + time_step_size

    velocity_primal_interface = mirror_function(
        velocity_primal_fluid, solid, fluid
    )
    velocity_primal_interface_before = mirror_function(
        velocity_primal_fluid_before, solid, fluid
    )
    return (
        param.RHO_SOLID
        * dot(velocity_primal_solid_before, first_test_function)
        * solid.dx
        + dot(displacement_primal_solid_before, second_test_function)
        * solid.dx
        - (1.0 - theta)
        * time_step_size
        * form_solid(
            velocity_primal_solid_before,
            displacement_primal_solid_before,
            first_test_function,
            second_test_function,
            solid,
            param,
        )
        + (1.0 - theta)
        * time_step_size
        * param.RHO_SOLID
        * characteristic_function_solid(param)
        * dot(external_force(time_before, param), first_test_function)
        * solid.dx
        + theta
        * time_step_size
        * param.RHO_SOLID
        * characteristic_function_solid(param)
        * dot(external_force(time, param), first_test_function)
        * solid.dx
    )


def functional_solid_interface(
    velocity_primal_solid_before,
    displacement_primal_solid_before,
    velocity_primal_fluid,
    displacement_primal_fluid,
    pressure_primal_fluid,
    velocity_primal_fluid_before,
    displacement_primal_fluid_before,
    pressure_primal_fluid_before,
    first_test_function,
    second_test_function,
    solid: Space,
    fluid: Space,
    param: Parameters,
    time,
    time_before,
    time_step_size,
    time_step_size_before,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
):

    theta = param.THETA + time_step_size

    return (
        -theta
        * time_step_size
        * form_solid_interface(
            velocity_primal_fluid,
            displacement_primal_fluid,
            pressure_primal_fluid,
            first_test_function,
            second_test_function,
            fluid,
            param,
            scheme=[True, False],
        )
        - (1.0 - theta)
        * time_step_size
        * form_solid_interface(
            velocity_primal_fluid_before,
            displacement_primal_fluid_before,
            pressure_primal_fluid_before,
            first_test_function,
            second_test_function,
            fluid,
            param,
            scheme=[True, False],
        )
        - time_step_size
        * form_solid_interface(
            velocity_primal_fluid,
            displacement_primal_fluid,
            pressure_primal_fluid,
            first_test_function,
            second_test_function,
            fluid,
            param,
            scheme=[False, True],
        )
    )


def form_solid_adjoint(
    velocity_solid_primal,
    displacement_solid_primal,
    velocity_solid_adjoint,
    displacement_solid_adjoint,
    first_test_function,
    second_test_function,
    solid: Space,
    param: Parameters,
    error_estimate=False,
):
    primal_functions = [velocity_solid_primal, displacement_solid_primal]
    adjoint_functions = [velocity_solid_adjoint, displacement_solid_adjoint]
    test_functions = [first_test_function, second_test_function]
    form_adjoint = create_adjoint(
        primal_functions,
        adjoint_functions,
        test_functions,
        form_solid,
        solid,
        param,
        False,
        False,
        error_estimate=error_estimate,
    )
    return form_adjoint


def form_solid_interface_adjoint(
    velocity_solid_primal,
    displacement_solid_primal,
    velocity_interface_adjoint,
    displacement_interface_adjoint,
    first_test_function,
    second_test_function,
    solid: Space,
    param: Parameters,
    error_estimate=False,
):
    primal_functions = [velocity_solid_primal, displacement_solid_primal]
    adjoint_functions = [
        velocity_interface_adjoint,
        displacement_interface_adjoint,
    ]
    test_functions = [first_test_function, second_test_function]
    form_adjoint = create_adjoint(
        primal_functions,
        adjoint_functions,
        test_functions,
        form_fluid_interface,
        solid,
        param,
        True,
        False,
        error_estimate=error_estimate,
    )
    return form_adjoint


def bilinear_form_solid_adjoint(
    velocity_adjoint_solid,
    displacement_adjoint_solid,
    first_test_function,
    second_test_function,
    solid: Space,
    param: Parameters,
    time_step_size,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    error_estimate=False,
):

    theta = param.THETA + time_step_size

    velocity_solid_primal = microtimestep.functions["primal_velocity"]
    displacement_solid_primal = microtimestep.functions["primal_displacement"]
    return (
        param.RHO_SOLID
        * dot(first_test_function, velocity_adjoint_solid)
        * solid.dx
        + dot(second_test_function, displacement_adjoint_solid) * solid.dx
        + theta
        * time_step_size
        * form_solid_adjoint(
            velocity_solid_primal,
            displacement_solid_primal,
            velocity_adjoint_solid,
            displacement_adjoint_solid,
            first_test_function,
            second_test_function,
            solid,
            param,
            error_estimate=error_estimate,
        )
    )


def functional_solid_adjoint(
    velocity_adjoint_solid_after,
    displacement_adjoint_solid_after,
    velocity_adjoint_fluid,
    displacement_adjoint_fluid,
    pressure_primal_fluid,
    velocity_adjoint_fluid_after,
    displacement_adjoint_fluid_after,
    pressure_primal_fluid_after,
    first_test_function,
    second_test_function,
    solid: Space,
    fluid: Space,
    param: Parameters,
    time,
    time_after,
    time_step_size,
    time_step_size_after,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
    initial,
    error_estimate=False,
):

    theta = param.THETA + time_step_size
    theta_after = param.THETA + time_step_size_after

    if initial:
        velocity_adjoint_solid_after = Function(solid.function_space_split[0])
        displacement_adjoint_solid_after = Function(
            solid.function_space_split[1]
        )
        velocity_adjoint_fluid_after = Function(fluid.function_space_split[0])
        displacement_adjoint_fluid_after = Function(
            fluid.function_space_split[1]
        )
        pressure_primal_fluid_after = Function(fluid.function_space_split[2])
    velocity_solid_primal = microtimestep.functions["primal_velocity"]
    displacement_solid_primal = microtimestep.functions["primal_displacement"]
    velocity_adjoint_interface = mirror_function(
        velocity_adjoint_fluid, solid, fluid
    )
    velocity_adjoint_interface_after = mirror_function(
        velocity_adjoint_fluid_after, solid, fluid
    )
    displacement_adjoint_interface = mirror_function(
        displacement_adjoint_fluid, solid, fluid
    )
    displacement_adjoint_interface_after = mirror_function(
        displacement_adjoint_fluid_after, solid, fluid
    )

    return (
        param.RHO_SOLID
        * dot(first_test_function, velocity_adjoint_solid_after)
        * solid.dx
        + dot(second_test_function, displacement_adjoint_solid_after)
        * solid.dx
        - (1.0 - theta_after)
        * time_step_size_after
        * form_solid_adjoint(
            velocity_solid_primal,
            displacement_solid_primal,
            velocity_adjoint_solid_after,
            displacement_adjoint_solid_after,
            first_test_function,
            second_test_function,
            solid,
            param,
            error_estimate=error_estimate,
        )
        + theta
        * time_step_size
        * form_solid_interface_adjoint(
            velocity_solid_primal,
            displacement_solid_primal,
            velocity_adjoint_interface,
            displacement_adjoint_interface,
            first_test_function,
            second_test_function,
            solid,
            param,
            error_estimate=error_estimate,
        )
        + (1.0 - theta_after)
        * time_step_size_after
        * form_solid_interface_adjoint(
            velocity_solid_primal,
            displacement_solid_primal,
            velocity_adjoint_interface_after,
            displacement_adjoint_interface_after,
            first_test_function,
            second_test_function,
            solid,
            param,
            error_estimate=error_estimate,
        )
        + 2.0
        * param.ZETA
        * characteristic_function_solid(param)
        * dot(
            goal_functional(
                microtimestep_before,
                microtimestep,
                "primal_displacement",
                solid,
            )[1],
            second_test_function,
        )
        * solid.dx
        + 2.0
        * param.ZETA
        * characteristic_function_solid(param)
        * dot(
            goal_functional(
                microtimestep,
                microtimestep_after,
                "primal_displacement",
                solid,
                initial,
            )[0],
            second_test_function,
        )
        * solid.dx
    )
