import numpy as np
from fenics import (
    Function,
    FunctionSpace,
    interpolate,
    project,
    inner,
    Constant,
    DirichletBC,
)
from solve_fluid import solve_fluid
from solve_solid import solve_solid
from scipy.sparse.linalg import LinearOperator, gmres
from parameters import Parameters
from spaces import Space
from initial import Initial
from coupling import mirror_function
from time_structure import MacroTimeStep

# Define shooting function
def shooting_function(
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

    # Save old values
    velocity_solid_new = Function(solid.function_space_split[0])
    velocity_solid_new.assign(velocity_solid.new)
    displacement_solid_new = Function(solid.function_space_split[1])
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
        solid_macrotimestep,
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
        fluid_macrotimestep,
        adjoint,
    )

    velocity_update = interpolate(
        project(
            velocity_solid_new - velocity_solid.new,
            solid.function_space_split[0],
        ),
        interface.function_space_split[0],
    )
    displacement_update = interpolate(
        project(
            displacement_solid_new - displacement_solid.new,
            solid.function_space_split[1],
        ),
        interface.function_space_split[1],
    )

    # Represent shooting function as an array
    velocity_update_array = velocity_update.vector().get_local()
    displacement_update_array = displacement_update.vector().get_local()
    shooting_function = np.concatenate(
        [velocity_update_array, displacement_update_array]
    )

    return shooting_function


# Define linear operator for linear solver in shooting method
def shooting_newton(
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
    velocity_interface,
    displacement_interface,
    shooting_function_value,
):
    def shooting_gmres(direction):

        # Define empty functions on interface
        increment_velocity = Function(interface.function_space_split[0])
        increment_displacement = Function(interface.function_space_split[1])

        # Set values of functions on interface
        direction_split = np.split(direction, 2)
        increment_velocity.vector().set_local(
            velocity_interface + param.EPSILON * direction_split[0]
        )
        increment_displacement.vector().set_local(
            displacement_interface + param.EPSILON * direction_split[1]
        )
        increment_velocity_solid = mirror_function(
            increment_velocity, solid, interface, True
        )
        increment_displacement_solid = mirror_function(
            increment_displacement, solid, interface, True
        )
        velocity_solid.new.assign(increment_velocity_solid)
        displacement_solid.new.assign(increment_displacement_solid)

        # Compute shooting function
        shooting_function_increment = shooting_function(
            velocity_fluid,
            displacement_fluid,
            pressure_fluid,
            velocity_solid,
            displacement_solid,
            first_time_step,
            fluid,
            solid,
            interface,
            param,
            fluid_macrotimestep,
            solid_macrotimestep,
            adjoint,
        )

        return (
            shooting_function_increment - shooting_function_value
        ) / param.EPSILON

    return shooting_gmres


def shooting(
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

    # Define initial values for Newton's method
    if solid_macrotimestep.before is not None:
        solid_macrotimestep_before_dt = solid_macrotimestep.before.dt
    else:
        solid_macrotimestep_before_dt = solid_macrotimestep.dt
    velocity_solid_new = Function(solid.function_space_split[0])
    velocity_solid_new.assign(
        project(
            velocity_solid.old
            + solid_macrotimestep.dt
            / solid_macrotimestep_before_dt
            * (velocity_solid.old - velocity_solid.old_old),
            solid.function_space_split[0],
        )
    )
    displacement_solid_new = Function(solid.function_space_split[1])
    displacement_solid_new.assign(
        project(
            displacement_solid.old
            + solid_macrotimestep.dt
            / solid_macrotimestep_before_dt
            * (displacement_solid.old - displacement_solid.old_old),
            solid.function_space_split[1],
        )
    )
    number_of_iterations = 0
    number_of_linear_systems = 0
    stop = False

    # Define Newton's method
    while not stop:

        number_of_iterations += 1
        number_of_linear_systems += 1
        print(f"Current iteration of Newton's method: {number_of_iterations}")

        # Define right hand side
        velocity_solid.new.assign(velocity_solid_new)
        displacement_solid.new.assign(displacement_solid_new)
        shooting_function_value = shooting_function(
            velocity_fluid,
            displacement_fluid,
            pressure_fluid,
            velocity_solid,
            displacement_solid,
            first_time_step,
            fluid,
            solid,
            interface,
            param,
            fluid_macrotimestep,
            solid_macrotimestep,
            adjoint,
        )
        shooting_function_value_linf = np.max(np.abs(shooting_function_value))
        if number_of_iterations == 1:

            if shooting_function_value_linf != 0.0:
                shooting_function_value_initial_linf = (
                    shooting_function_value_linf
                )
            else:
                shooting_function_value_initial_linf = 1.0

        print(
            f"Absolute error on the interface in infinity norm: "
            f"{shooting_function_value_linf}"
        )
        print(
            f"Relative error on the interface in infinity norm: "
            f"{shooting_function_value_linf / shooting_function_value_initial_linf}"
        )

        # Check stop conditions
        if (
            shooting_function_value_linf < param.ABSOLUTE_TOLERANCE_NEWTON
            or shooting_function_value_linf
            / shooting_function_value_initial_linf
            < param.RELATIVE_TOLERANCE_NEWTON
        ):
            print(
                f"Newton's method converged successfully after "
                f"{number_of_iterations} iterations and solving "
                f"{number_of_linear_systems} linear systems."
            )
            stop = True

        elif number_of_iterations == param.MAX_ITERATIONS_NEWTON:

            print("Newton's method failed to converge.")
            stop = True
            number_of_linear_systems = -1

        if not stop:

            # Define linear operator
            velocity_solid_interface = interpolate(
                velocity_solid_new, interface.function_space_split[0]
            )
            velocity_solid_interface_array = (
                velocity_solid_interface.vector().get_local()
            )
            displacement_solid_interface = interpolate(
                displacement_solid_new, interface.function_space_split[1]
            )
            displacement_solid_interface_array = (
                displacement_solid_interface.vector().get_local()
            )
            linear_operator_newton = shooting_newton(
                velocity_fluid,
                displacement_fluid,
                pressure_fluid,
                velocity_solid,
                displacement_solid,
                first_time_step,
                fluid,
                solid,
                interface,
                param,
                fluid_macrotimestep,
                solid_macrotimestep,
                adjoint,
                velocity_solid_interface_array,
                displacement_solid_interface_array,
                shooting_function_value,
            )
            operator_size = len(
                2 * (fluid.interface_table[0] + fluid.interface_table[1])
            )
            shooting_gmres = LinearOperator(
                (operator_size, operator_size), matvec=linear_operator_newton
            )

            # Solve linear system
            number_of_iterations_gmres = 0

            def callback(vector):

                nonlocal number_of_iterations_gmres
                global residual_norm_gmres
                number_of_iterations_gmres += 1
                print(
                    f"Current iteration of GMRES method: {number_of_iterations_gmres}"
                )
                residual_norm_gmres = np.linalg.norm(vector)

            if not adjoint:
                param.TOLERANCE_GMRES = max(
                    shooting_function_value_linf,
                    param.ABSOLUTE_TOLERANCE_NEWTON,
                )
                param.EPSILON = shooting_function_value_linf
            direction, exit_code = gmres(
                shooting_gmres,
                -shooting_function_value,
                tol=param.TOLERANCE_GMRES,
                restart=30,
                maxiter=param.MAX_ITERATIONS_GMRES,
                callback=callback,
            )
            number_of_linear_systems += number_of_iterations_gmres
            if exit_code == 0:

                print(
                    f"GMRES method converged successfully after "
                    f"{number_of_iterations_gmres} iterations"
                )

            else:

                print("GMRES method failed to converge.")
                print(f"Norm of residual: {residual_norm_gmres}")

            # Advance solution
            direction_split = np.split(direction, 2)
            velocity_solid_interface_array += direction_split[0]
            displacement_solid_interface_array += direction_split[1]
            velocity_solid_interface.vector().set_local(
                velocity_solid_interface_array
            )
            displacement_solid_interface.vector().set_local(
                displacement_solid_interface_array
            )
            velocity_solid_new.assign(
                mirror_function(
                    velocity_solid_interface, solid, interface, True
                )
            )
            displacement_solid_new.assign(
                mirror_function(
                    displacement_solid_interface, solid, interface, True
                )
            )

    velocity_fluid.iterations.append(number_of_linear_systems)
    displacement_fluid.iterations.append(number_of_linear_systems)
    pressure_fluid.iterations.append(number_of_linear_systems)
    velocity_solid.iterations.append(number_of_linear_systems)
    displacement_solid.iterations.append(number_of_linear_systems)

    return
