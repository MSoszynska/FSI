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
    assemble_system,
    norm,
    dot,
    inv,
)
from spaces import Space
from parameters import Parameters
from time_structure import MacroTimeStep
from initial import Initial
from coupling import mirror_form
from forms import (
    bilinear_form_fluid_correction,
    functional_fluid_correction,
    sigma_solid,
)

# Define a function solving a problem on a subdomain
def solve_solid(
    velocity_solid: Initial,
    displacement_solid: Initial,
    velocity_fluid: Initial,
    displacement_fluid: Initial,
    pressure_fluid: Initial,
    solid: Space,
    fluid: Space,
    first_time_step,
    param: Parameters,
    macrotimestep_solid: MacroTimeStep,
    macrotimestep_fluid: MacroTimeStep,
    adjoint,
    save=False,
):

    # Store old solutions
    velocity_solid_old = Function(solid.function_space_split[0])
    displacement_solid_old = Function(solid.function_space_split[1])
    velocity_solid_old.assign(velocity_solid.old)
    displacement_solid_old.assign(displacement_solid.old)

    # Store old interface values
    velocity_old_interface = Function(fluid.function_space_split[0])
    displacement_old_interface = Function(fluid.function_space_split[1])
    pressure_old_interface = Function(fluid.function_space_split[2])
    velocity_old_interface.assign(velocity_fluid.old_average)
    displacement_old_interface.assign(displacement_fluid.old_average)
    pressure_old_interface.assign(pressure_fluid.old_average)

    # Initialize new interface values
    velocity_interface = Function(fluid.function_space_split[0])
    displacement_interface = Function(fluid.function_space_split[1])
    pressure_interface = Function(fluid.function_space_split[2])

    # Define time pointers
    if adjoint:

        microtimestep = macrotimestep_solid.tail.before

    else:

        microtimestep = macrotimestep_solid.head

    # Compute macro time-step size
    size = macrotimestep_solid.size - 1
    for m in range(size):

        # Initialize average values
        velocity_solid_average_temp = Function(solid.function_space_split[0])
        displacement_solid_average_temp = Function(
            solid.function_space_split[1]
        )
        velocity_solid_average = Function(solid.function_space_split[0])
        displacement_solid_average = Function(solid.function_space_split[1])

        # Extrapolate weak boundary conditions on the interface
        if adjoint:

            extrapolation_proportion = (
                microtimestep.point - macrotimestep_solid.head.point
            ) / macrotimestep_solid.dt
            time_step_size = microtimestep.dt
            microtimestep_form = microtimestep.after
            microtimestep_form_before = microtimestep
            if m == 0 and macrotimestep_solid.after is None:
                time_step_size_old = microtimestep.dt
                microtimestep_form_after = microtimestep_form
            elif m == 0:
                time_step_size_old = (
                    macrotimestep_solid.microtimestep_after.before.dt
                )
                microtimestep_form_after = (
                    macrotimestep_solid.microtimestep_after
                )
            else:
                time_step_size_old = microtimestep.after.dt
                microtimestep_form_after = microtimestep_form.after

        else:

            extrapolation_proportion = (
                macrotimestep_solid.tail.point - microtimestep.after.point
            ) / macrotimestep_solid.dt
            time_step_size = microtimestep.dt
            time_step_size_old = microtimestep.dt
            microtimestep_form_before = None
            microtimestep_form = None
            microtimestep_form_after = None

        # Define intermediate solutions
        velocity_interface.assign(
            project(
                extrapolation_proportion * velocity_fluid.old
                + (1.0 - extrapolation_proportion) * velocity_fluid.new,
                fluid.function_space_split[0],
            )
        )
        displacement_interface.assign(
            project(
                extrapolation_proportion * displacement_fluid.old
                + (1.0 - extrapolation_proportion) * displacement_fluid.new,
                fluid.function_space_split[1],
            )
        )
        pressure_interface.assign(
            project(
                pressure_fluid.new_average,
                fluid.function_space_split[2],
            )
        )

        # Define correction scheme
        if not adjoint and macrotimestep_solid.size > 2 and m < size - 1:

            # Define trial and test functions
            trial_function_correction = TrialFunction(
                fluid.function_space_correction
            )
            (velocity_fluid_correction, pressure_fluid_correction) = split(
                trial_function_correction
            )
            test_function_correction = TestFunction(
                fluid.function_space_correction
            )
            (
                first_test_function_correction,
                second_test_function_correction,
            ) = split(test_function_correction)

            # Define scheme
            left_hand_side_correction = bilinear_form_fluid_correction(
                velocity_fluid_correction,
                pressure_fluid_correction,
                velocity_interface,
                displacement_interface,
                pressure_interface,
                first_test_function_correction,
                second_test_function_correction,
                fluid,
                param,
            )
            right_hand_side_correction = functional_fluid_correction(
                velocity_interface,
                displacement_interface,
                pressure_interface,
                first_test_function_correction,
                second_test_function_correction,
                fluid,
                param,
            )

            # Solve correction problem
            trial_function_correction = Function(
                fluid.function_space_correction
            )
            solve(
                left_hand_side_correction == right_hand_side_correction,
                trial_function_correction,
                fluid.boundaries(
                    microtimestep.after.point,
                    param,
                    adjoint,
                    correction_space=True,
                ),
            )
            (
                velocity_fluid_correction,
                pressure_fluid_correction,
            ) = trial_function_correction.split(trial_function_correction)

            # Assign new interface values
            velocity_interface.assign(velocity_fluid_correction)
            pressure_interface.assign(pressure_fluid_correction)

        # Define trial and test functions
        trial_function = TrialFunction(solid.function_space)
        (velocity_solid_new, displacement_solid_new) = split(trial_function)
        test_function = TestFunction(solid.function_space)
        (first_test_function, second_test_function) = split(test_function)

        # Define scheme
        time = microtimestep.after.point
        time_before = microtimestep.point
        initial = False
        if not adjoint:
            bilinear_form = solid.primal_problem.bilinear_form
            functional = solid.primal_problem.functional
            functional_interface = solid.primal_problem.functional_interface
        else:
            bilinear_form = solid.adjoint_problem.bilinear_form
            functional = solid.adjoint_problem.functional
            if first_time_step and m == 0:
                initial = True
        left_hand_side = bilinear_form(
            velocity_solid_new,
            displacement_solid_new,
            first_test_function,
            second_test_function,
            solid,
            param,
            time_step_size,
            microtimestep_form_before,
            microtimestep_form,
        )
        right_hand_side = functional(
            velocity_solid_old,
            displacement_solid_old,
            velocity_interface,
            displacement_interface,
            pressure_interface,
            velocity_old_interface,
            displacement_old_interface,
            pressure_old_interface,
            first_test_function,
            second_test_function,
            solid,
            fluid,
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
        if not adjoint:
            test_function_interface = TestFunction(fluid.function_space)
            (
                first_test_function_interface,
                second_test_function_interface,
                third_test_function_interface,
            ) = split(test_function_interface)
            right_hand_side_interface = functional_interface(
                velocity_solid_old,
                displacement_solid_old,
                velocity_interface,
                displacement_interface,
                pressure_interface,
                velocity_old_interface,
                displacement_old_interface,
                pressure_old_interface,
                first_test_function_interface,
                second_test_function_interface,
                solid,
                fluid,
                param,
                time,
                time_before,
                time_step_size,
                time_step_size_old,
                microtimestep_form_before,
                microtimestep_form,
                microtimestep_form_after,
            )
            right_hand_side_interface_assemble = assemble(
                right_hand_side_interface
            )
            right_hand_side_interface_vector = (
                right_hand_side_interface_assemble.get_local()
            )
            right_hand_side_interface_mirror = mirror_form(
                right_hand_side_interface_vector, solid, fluid
            )
            right_hand_side_assemble += right_hand_side_interface_mirror

        # Solve problem
        time = microtimestep.after.point
        if adjoint:
            left_hand_side_assemble = assemble(left_hand_side)
            trial_function = Function(solid.function_space)
            [
                boundary.apply(
                    left_hand_side_assemble, right_hand_side_assemble
                )
                for boundary in solid.boundaries(time, param, adjoint)
            ]
            solve(
                left_hand_side_assemble,
                trial_function.vector(),
                right_hand_side_assemble,
            )
            (
                velocity_solid_new,
                displacement_solid_new,
            ) = trial_function.split(trial_function)
        else:

            trial_function_new = Function(solid.function_space)
            form = left_hand_side - right_hand_side
            form = action(form, trial_function_new)
            jacobian = derivative(form, trial_function_new, trial_function)
            boundaries = solid.boundaries(time, param, adjoint)
            boundaries_homogenize = boundaries
            [boundary.homogenize() for boundary in boundaries_homogenize]
            solution_update = Function(solid.function_space)
            iterations = 0
            epsilon = 1.0

            while epsilon > 1.0e-10 and iterations < 30:
                iterations += 1
                jacobian_assemble = assemble(jacobian)
                form_assemble = assemble(-form)
                form_assemble += right_hand_side_interface_mirror
                [
                    boundary.apply(jacobian_assemble, form_assemble)
                    for boundary in boundaries_homogenize
                ]
                solve(
                    jacobian_assemble, solution_update.vector(), form_assemble
                )
                epsilon = norm(solution_update, "L2")
                trial_function_new.vector()[:] += solution_update.vector()
            print(f"Newton solver finished in {iterations} iterations")
            (
                velocity_solid_new,
                displacement_solid_new,
            ) = trial_function_new.split(trial_function)

        # Save solutions
        if save:

            velocity_solid.save(velocity_solid_new)
            displacement_solid.save(displacement_solid_new)

        # Update average values
        velocity_solid_average_temp.assign(velocity_solid_average)
        displacement_solid_average_temp.assign(displacement_solid_average)
        # velocity_solid_average.assign(project(velocity_solid_average_temp
        #                                       + 0.5 * time_step_size / macrotimestep_solid.dt
        #                                       * (velocity_solid_old + velocity_solid_new),
        #                                       solid.function_space_split[0]))
        # displacement_solid_average.assign(project(displacement_solid_average_temp
        #                                           + 0.5 * time_step_size / macrotimestep_solid.dt
        #                                           * (displacement_solid_old + displacement_solid_new),
        #                                       solid.function_space_split[1]))
        velocity_solid_average.assign(velocity_solid_new)
        displacement_solid_average.assign(displacement_solid_new)

        # Update solution
        velocity_solid_old.assign(velocity_solid_new)
        displacement_solid_old.assign(displacement_solid_new)

        # Update boundary conditions
        velocity_old_interface.assign(velocity_interface)
        displacement_old_interface.assign(displacement_interface)
        pressure_old_interface.assign(pressure_interface)

        # Advance timeline
        if adjoint:

            microtimestep = microtimestep.before

        else:

            microtimestep = microtimestep.after

    # Save final values
    # velocity_solid_average.assign(velocity_solid_new)
    # displacement_solid_average.assign(displacement_solid_new)

    velocity_solid.new.assign(velocity_solid_new)
    displacement_solid.new.assign(displacement_solid_new)
    velocity_solid.new_average.assign(velocity_solid_average)
    displacement_solid.new_average.assign(displacement_solid_average)

    return
