from fenics import (
    Function,
    FunctionSpace,
    project,
    DirichletBC,
    Constant,
    TrialFunction,
    split,
    TestFunction,
    solve,
    assemble,
    Expression,
    assign,
    action,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    info,
    det,
    inv,
    dot,
)
from spaces import Space
from parameters import Parameters
from time_structure import MacroTimeStep
from initial import Initial
from forms import deformation_gradient, sigma_fluid

# Define a function solving a problem on a subdomain
def solve_fluid(
    velocity_fluid: Initial,
    displacement_fluid: Initial,
    pressure_fluid: Initial,
    velocity_solid: Initial,
    displacement_solid: Initial,
    fluid: Space,
    solid: Space,
    first_time_step,
    param: Parameters,
    macrotimestep_fluid: MacroTimeStep,
    macrotimestep_solid: MacroTimeStep,
    adjoint,
    save=False,
):

    # Store old solutions
    velocity_fluid_old = Function(fluid.function_space_split[0])
    displacement_fluid_old = Function(fluid.function_space_split[1])
    pressure_fluid_old = Function(fluid.function_space_split[2])
    velocity_fluid_old.assign(velocity_fluid.old)
    displacement_fluid_old.assign(displacement_fluid.old)
    pressure_fluid_old.assign(pressure_fluid.old)

    # Store old interface values
    velocity_old_interface = Function(solid.function_space_split[0])
    displacement_old_interface = Function(solid.function_space_split[1])
    velocity_old_interface.assign(velocity_solid.old_average)
    displacement_old_interface.assign(displacement_solid.old_average)

    # Initialize new interface values
    velocity_interface = Function(solid.function_space_split[0])
    displacement_interface = Function(solid.function_space_split[1])

    # Define time pointers
    if adjoint:

        microtimestep = macrotimestep_fluid.tail.before

    else:

        microtimestep = macrotimestep_fluid.head

    # Compute macro time-step size
    size = macrotimestep_fluid.size - 1
    for m in range(size):

        # Initialize average values
        velocity_fluid_average_temp = Function(fluid.function_space_split[0])
        displacement_fluid_average_temp = Function(
            fluid.function_space_split[1]
        )
        pressure_fluid_average_temp = Function(fluid.function_space_split[2])
        velocity_fluid_average = Function(fluid.function_space_split[0])
        displacement_fluid_average = Function(fluid.function_space_split[1])
        pressure_fluid_average = Function(fluid.function_space_split[2])

        # Extrapolate weak boundary conditions on the interface
        if adjoint:

            extrapolation_proportion = (
                microtimestep.point - macrotimestep_fluid.head.point
            ) / macrotimestep_fluid.dt
            time_step_size = microtimestep.dt
            microtimestep_form = microtimestep.after
            microtimestep_form_before = microtimestep
            if m == 0 and macrotimestep_fluid.after is None:
                time_step_size_old = microtimestep.dt
                microtimestep_form_after = microtimestep_form
            elif m == 0:
                time_step_size_old = (
                    macrotimestep_fluid.microtimestep_after.before.dt
                )
                microtimestep_form_after = (
                    macrotimestep_fluid.microtimestep_after
                )
            else:
                time_step_size_old = microtimestep.after.dt
                microtimestep_form_after = microtimestep_form.after

        else:

            extrapolation_proportion = (
                macrotimestep_fluid.tail.point - microtimestep.after.point
            ) / macrotimestep_fluid.dt
            time_step_size = microtimestep.dt
            time_step_size_old = microtimestep.dt
            microtimestep_form_before = None
            microtimestep_form = None
            microtimestep_form_after = None

        # Define intermediate solutions
        velocity_interface.assign(
            project(
                extrapolation_proportion * velocity_solid.old
                + (1.0 - extrapolation_proportion) * velocity_solid.new,
                solid.function_space_split[0],
            )
        )
        displacement_interface.assign(
            project(
                extrapolation_proportion * displacement_solid.old
                + (1.0 - extrapolation_proportion) * displacement_solid.new,
                solid.function_space_split[1],
            )
        )

        # Define trial and test functions
        trial_function = TrialFunction(fluid.function_space)
        (
            velocity_fluid_new,
            displacement_fluid_new,
            pressure_fluid_new,
        ) = split(trial_function)
        test_function = TestFunction(fluid.function_space)
        (
            first_test_function,
            second_test_function,
            third_test_function,
        ) = split(test_function)

        # Define scheme
        time = microtimestep.after.point
        time_before = microtimestep.point
        initial = False
        if not adjoint:
            bilinear_form = fluid.primal_problem.bilinear_form
            functional = fluid.primal_problem.functional
        else:
            bilinear_form = fluid.adjoint_problem.bilinear_form
            functional = fluid.adjoint_problem.functional
            if first_time_step and m == 0:
                initial = True
        left_hand_side = bilinear_form(
            velocity_fluid_new,
            displacement_fluid_new,
            pressure_fluid_new,
            first_test_function,
            second_test_function,
            third_test_function,
            velocity_fluid_old,
            displacement_fluid_old,
            pressure_fluid_old,
            fluid,
            param,
            time_step_size,
            microtimestep_form_before,
            microtimestep_form,
        )
        right_hand_side = functional(
            velocity_fluid_old,
            displacement_fluid_old,
            pressure_fluid_old,
            velocity_interface,
            displacement_interface,
            velocity_old_interface,
            displacement_old_interface,
            first_test_function,
            second_test_function,
            third_test_function,
            fluid,
            solid,
            param,
            time,
            time_before,
            time_step_size,
            time_step_size_old,
            microtimestep_form_before,
            microtimestep_form,
            microtimestep_form_after,
            initial,
        )
        right_hand_side_assemble = assemble(right_hand_side)

        # Solve problem
        time = microtimestep.after.point
        if adjoint:
            left_hand_side_assemble = assemble(left_hand_side)
            trial_function = Function(fluid.function_space)
            [
                boundary.apply(
                    left_hand_side_assemble, right_hand_side_assemble
                )
                for boundary in fluid.boundaries(time, param, adjoint)
            ]
            solve(
                left_hand_side_assemble,
                trial_function.vector(),
                right_hand_side_assemble,
            )
            (
                velocity_fluid_new,
                displacement_fluid_new,
                pressure_fluid_new,
            ) = trial_function.split(trial_function)
        else:
            trial_function_new = Function(fluid.function_space)
            form = left_hand_side - right_hand_side
            form = action(form, trial_function_new)
            jacobian = derivative(form, trial_function_new, trial_function)
            boundaries = fluid.boundaries(time, param, adjoint)
            problem = NonlinearVariationalProblem(
                form, trial_function_new, boundaries, jacobian
            )
            solver = NonlinearVariationalSolver(problem)
            # prm = solver.parameters
            # info(prm, True)
            solver.parameters["newton_solver"]["report"] = False
            # solver.parameters["newton_solver"]["maximum_iterations"] = 100
            solver.solve()
            (
                velocity_fluid_new,
                displacement_fluid_new,
                pressure_fluid_new,
            ) = trial_function_new.split(trial_function)

        # Save solutions
        if save:
            velocity_fluid.save(velocity_fluid_new)
            displacement_fluid.save(displacement_fluid_new)
            pressure_fluid.save(pressure_fluid_new)

        # Update average values
        velocity_fluid_average_temp.assign(velocity_fluid_average)
        displacement_fluid_average_temp.assign(displacement_fluid_average)
        pressure_fluid_average_temp.assign(pressure_fluid_average)
        # velocity_fluid_average.assign(project(velocity_fluid_average_temp
        #                                       + 0.5 * time_step_size / macrotimestep_fluid.dt
        #                                       * (velocity_fluid_old + velocity_fluid_new),
        #                                       fluid.function_space_split[0]))
        # displacement_fluid_average.assign(project(displacement_fluid_average_temp
        #                                           + 0.5 * time_step_size / macrotimestep_fluid.dt
        #                                           * (displacement_fluid_old + displacement_fluid_new),
        #                                       fluid.function_space_split[1]))
        # pressure_fluid_average.assign(project(pressure_fluid_average_temp + time_step_size / macrotimestep_fluid.dt * pressure_fluid_new,
        #                                       fluid.function_space_split[2]))
        velocity_fluid_average.assign(velocity_fluid_new)
        displacement_fluid_average.assign(displacement_fluid_new)
        pressure_fluid_average.assign(pressure_fluid_new)

        # Update solution
        velocity_fluid_old.assign(velocity_fluid_new)
        displacement_fluid_old.assign(displacement_fluid_new)
        pressure_fluid_old.assign(pressure_fluid_new)

        # Update boundary conditions
        velocity_old_interface.assign(velocity_interface)
        displacement_old_interface.assign(displacement_interface)

        # Advance timeline
        if adjoint:

            microtimestep = microtimestep.before

        else:

            microtimestep = microtimestep.after

    # Save final values
    # velocity_fluid_average.assign(velocity_fluid_new)
    # displacement_fluid_average.assign(displacement_fluid_new)
    # pressure_fluid_average.assign(pressure_fluid_new)

    velocity_fluid.new.assign(velocity_fluid_new)
    displacement_fluid.new.assign(displacement_fluid_new)
    pressure_fluid.new.assign(pressure_fluid_new)
    velocity_fluid.new_average.assign(velocity_fluid_average)
    displacement_fluid.new_average.assign(displacement_fluid_average)
    pressure_fluid.new_average.assign(pressure_fluid_average)

    return
