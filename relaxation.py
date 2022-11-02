from fenics import Function, FunctionSpace, project, interpolate, norm
from solve_fluid import solve_fluid
from solve_solid import solve_solid
from parameters import Parameters
from spaces import Space
from time_structure import MacroTimeStep
from initial import Initial

# Define relaxation method
def relaxation(
    velocity_fluid: Initial,
    displacement_fluid: Initial,
    pressure_fluid: Initial,
    velocity_solid: Initial,
    displacement_solid: Initial,
    first_time_step,
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    fluid_macrotimestep: MacroTimeStep,
    solid_macrotimestep: MacroTimeStep,
    adjoint,
):

    # Define initial values for relaxation method
    velocity_solid_new = Function(solid.function_space_split[0])
    displacement_solid_new = Function(solid.function_space_split[1])
    number_of_iterations = 0
    stop = False

    while not stop:

        number_of_iterations += 1
        print(
            f"Current iteration of relaxation method: {number_of_iterations}"
        )

        # Save old values
        velocity_solid_new.assign(velocity_solid.new)
        displacement_solid_new.assign(displacement_solid.new)

        # Perform one iteration
        solve_fluid(
            velocity_fluid,
            displacement_fluid,
            pressure_fluid,
            velocity_solid,
            displacement_solid,
            fluid,
            solid,
            first_time_step,
            param,
            fluid_macrotimestep,
            adjoint,
        )
        solve_solid(
            velocity_solid,
            displacement_solid,
            velocity_fluid,
            displacement_fluid,
            pressure_fluid,
            solid,
            fluid,
            first_time_step,
            param,
            solid_macrotimestep,
            adjoint,
        )

        # Perform relaxation
        velocity_solid.new.assign(
            project(
                param.TAU * velocity_solid.new
                + (1.0 - param.TAU) * velocity_solid_new,
                solid.function_space_split[0],
            )
        )
        displacement_solid.new.assign(
            project(
                param.TAU * displacement_solid.new
                + (1.0 - param.TAU) * displacement_solid_new,
                solid.function_space_split[1],
            )
        )

        # Define errors on the interface
        velocity_error = interpolate(
            project(
                velocity_solid_new - velocity_solid.new,
                solid.function_space_split[0],
            ),
            interface.function_space_split[0],
        )
        velocity_error_linf = norm(velocity_error.vector(), "linf")
        displacement_error = interpolate(
            project(
                displacement_solid_new - displacement_solid.new,
                solid.function_space_split[1],
            ),
            interface.function_space_split[1],
        )
        displacement_error_linf = norm(displacement_error.vector(), "linf")
        error_linf = max(velocity_error_linf, displacement_error_linf)
        if number_of_iterations == 1:

            if error_linf != 0.0:
                error_initial_linf = error_linf
            else:
                error_initial_linf = 1.0

        print(f"Absolute error on the interface: {error_linf}")
        print(
            f"Relative error on the interface: {error_linf / error_initial_linf}"
        )

        # Check stop conditions
        if (
            error_linf < param.ABSOLUTE_TOLERANCE_RELAXATION
            or error_linf / error_initial_linf
            < param.RELATIVE_TOLERANCE_RELAXATION
        ):

            print(
                f"Algorithm converged successfully after "
                f"{number_of_iterations} iterations"
            )
            stop = True

        elif number_of_iterations == param.MAX_ITERATIONS_RELAXATION:

            print("Maximal number of iterations was reached.")
            stop = True
            number_of_iterations = -1

    velocity_fluid.iterations.append(number_of_iterations)
    displacement_fluid.iterations.append(number_of_iterations)
    pressure_fluid.iterations.append(number_of_iterations)
    velocity_solid.iterations.append(number_of_iterations)
    displacement_solid.iterations.append(number_of_iterations)

    return
