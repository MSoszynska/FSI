from fenics import (
    Function,
    UserExpression,
    DirichletBC,
    Constant,
    assemble,
    inner,
    Expression,
    project,
    File,
    dot,
    TestFunction,
    split,
    det,
    inv,
    assign,
    ALE,
)
from solve_fluid import solve_fluid
from solve_solid import solve_solid
from initial import Initial
from spaces import Space
from parameters import Parameters
from time_structure import TimeLine
from coupling import mirror_function, mirror_form
from forms import deformation_gradient, sigma_fluid, sigma_solid


def time_stepping(
    fluid: Space,
    solid: Space,
    interface: Space,
    param: Parameters,
    decoupling,
    fluid_timeline: TimeLine,
    solid_timeline: TimeLine,
    adjoint,
):

    # Initialize function objects
    if adjoint:

        velocity_name = "adjoint_velocity"
        displacement_name = "adjoint_displacement"
        pressure_name = "adjoint_pressure"

    else:

        velocity_name = "primal_velocity"
        displacement_name = "primal_displacement"
        pressure_name = "primal_pressure"

    velocity_fluid = Initial(
        "fluid", velocity_name, fluid.function_space_split[0]
    )
    displacement_fluid = Initial(
        "fluid", displacement_name, fluid.function_space_split[1]
    )
    pressure_fluid = Initial(
        "fluid", pressure_name, fluid.function_space_split[2]
    )
    velocity_solid = Initial(
        "solid", velocity_name, solid.function_space_split[0]
    )
    displacement_solid = Initial(
        "solid", displacement_name, solid.function_space_split[1]
    )

    # Save initial values for the primal problem
    if not adjoint and param.INITIAL_TIME == 0.0 and param.PARTIAL_LOAD == 0:

        velocity_fluid.save(Function(fluid.function_space_split[0]))
        displacement_fluid.save(Function(fluid.function_space_split[1]))
        pressure_fluid.save(Function(fluid.function_space_split[2]))
        velocity_solid.save(Function(solid.function_space_split[0]))
        displacement_solid.save(Function(solid.function_space_split[1]))

    # Define time pointers
    if adjoint:

        fluid_macrotimestep = fluid_timeline.tail
        solid_macrotimestep = solid_timeline.tail

    else:

        fluid_macrotimestep = fluid_timeline.head
        solid_macrotimestep = solid_timeline.head

    # Partially load solutions
    if not adjoint and param.PARTIAL_LOAD > 0:
        velocity_fluid.save(velocity_fluid.load(0), True)
        displacement_fluid.save(displacement_fluid.load(0), True)
        pressure_fluid.save(pressure_fluid.load(0), True)
        velocity_solid.save(velocity_solid.load(0), True)
        displacement_solid.save(displacement_solid.load(0), True)
        temp_fluid_macrotimestep = fluid_macrotimestep
        temp_solid_macrotimestep = solid_macrotimestep
        temp_fluid_microtimestep = temp_fluid_macrotimestep.head.after
        temp_solid_microtimestep = temp_solid_macrotimestep.head.after
        fluid_counter = 0
        solid_counter = 0
        left_load = param.PARTIAL_LOAD
        stop_load = False
        while not stop_load:
            if (temp_fluid_microtimestep is None) and (
                temp_solid_microtimestep is None
            ):
                left_load -= 1
                if left_load == 0:
                    stop_load = True
                temp_fluid_macrotimestep = temp_fluid_macrotimestep.after
                temp_solid_macrotimestep = temp_solid_macrotimestep.after
                temp_fluid_microtimestep = temp_fluid_macrotimestep.head.after
                temp_solid_microtimestep = temp_solid_macrotimestep.head.after
            if temp_fluid_microtimestep is not None and not stop_load:
                temp_fluid_microtimestep = temp_fluid_microtimestep.after
                fluid_counter += 1
                velocity_fluid.save(velocity_fluid.load(fluid_counter), True)
                displacement_fluid.save(
                    displacement_fluid.load(fluid_counter), True
                )
                pressure_fluid.save(pressure_fluid.load(fluid_counter), True)
            if temp_solid_microtimestep is not None and not stop_load:
                temp_solid_microtimestep = temp_solid_microtimestep.after
                solid_counter += 1
                velocity_solid.save(velocity_solid.load(solid_counter), True)
                displacement_solid.save(
                    displacement_solid.load(solid_counter), True
                )
        fluid_macrotimestep = temp_fluid_macrotimestep
        solid_macrotimestep = temp_solid_macrotimestep
        velocity_fluid.old_old = velocity_fluid.load(fluid_counter)
        displacement_fluid.old_old = displacement_fluid.load(fluid_counter)
        pressure_fluid.old_old = pressure_fluid.load(fluid_counter)
        velocity_solid.old_old = velocity_solid.load(solid_counter)
        displacement_solid.old_old = displacement_solid.load(solid_counter)
        velocity_fluid.old = velocity_fluid.load(fluid_counter)
        displacement_fluid.old = displacement_fluid.load(fluid_counter)
        pressure_fluid.old = pressure_fluid.load(fluid_counter)
        velocity_solid.old = velocity_solid.load(solid_counter)
        displacement_solid.old = displacement_solid.load(solid_counter)
        velocity_fluid.new = velocity_fluid.load(fluid_counter)
        displacement_fluid.new = displacement_fluid.load(fluid_counter)
        pressure_fluid.new = pressure_fluid.load(fluid_counter)
        velocity_solid.new = velocity_solid.load(solid_counter)
        displacement_solid.new = displacement_solid.load(solid_counter)

        velocity_fluid.old_old_average = velocity_fluid.load(fluid_counter)
        displacement_fluid.old_old_average = displacement_fluid.load(
            fluid_counter
        )
        pressure_fluid.old_old_average = pressure_fluid.load(fluid_counter)
        velocity_solid.old_old_average = velocity_solid.load(solid_counter)
        displacement_solid.old_old_average = displacement_solid.load(
            solid_counter
        )
        velocity_fluid.old_average = velocity_fluid.load(fluid_counter)
        displacement_fluid.old_average = displacement_fluid.load(fluid_counter)
        pressure_fluid.old_average = pressure_fluid.load(fluid_counter)
        velocity_solid.old_average = velocity_solid.load(solid_counter)
        displacement_solid.old_average = displacement_solid.load(solid_counter)
        velocity_fluid.new_average = velocity_fluid.load(fluid_counter)
        displacement_fluid.new_average = displacement_fluid.load(fluid_counter)
        pressure_fluid.new_average = pressure_fluid.load(fluid_counter)
        velocity_solid.new_average = velocity_solid.load(solid_counter)
        displacement_solid.new_average = displacement_solid.load(solid_counter)

        velocity_fluid.HDF5_counter = fluid_counter + 1
        displacement_fluid.HDF5_counter = fluid_counter + 1
        pressure_fluid.HDF5_counter = fluid_counter + 1
        velocity_solid.HDF5_counter = solid_counter + 1
        displacement_solid.HDF5_counter = solid_counter + 1

    if not adjoint and param.INITIAL_TIME > 0.0 and param.PARTIAL_LOAD == 0:
        counter = param.INITIAL_COUNTER
        velocity_fluid.old_old = velocity_fluid.load(counter)
        displacement_fluid.old_old = displacement_fluid.load(counter)
        pressure_fluid.old_old = pressure_fluid.load(counter)
        velocity_solid.old_old = velocity_solid.load(counter)
        displacement_solid.old_old = displacement_solid.load(counter)
        velocity_fluid.old = velocity_fluid.load(counter)
        displacement_fluid.old = displacement_fluid.load(counter)
        pressure_fluid.old = pressure_fluid.load(counter)
        velocity_solid.old = velocity_solid.load(counter)
        displacement_solid.old = displacement_solid.load(counter)
        velocity_fluid.new = velocity_fluid.load(counter)
        displacement_fluid.new = displacement_fluid.load(counter)
        pressure_fluid.new = pressure_fluid.load(counter)
        velocity_solid.new = velocity_solid.load(counter)
        displacement_solid.new = displacement_solid.load(counter)

        velocity_fluid.old_old_average = velocity_fluid.load(counter)
        displacement_fluid.old_old_average = displacement_fluid.load(counter)
        pressure_fluid.old_old_average = pressure_fluid.load(counter)
        velocity_solid.old_old_average = velocity_solid.load(counter)
        displacement_solid.old_old_average = displacement_solid.load(counter)
        velocity_fluid.old_average = velocity_fluid.load(counter)
        displacement_fluid.old_average = displacement_fluid.load(counter)
        pressure_fluid.old_average = pressure_fluid.load(counter)
        velocity_solid.old_average = velocity_solid.load(counter)
        displacement_solid.old_average = displacement_solid.load(counter)
        velocity_fluid.new_average = velocity_fluid.load(counter)
        displacement_fluid.new_average = displacement_fluid.load(counter)
        pressure_fluid.new_average = pressure_fluid.load(counter)
        velocity_solid.new_average = velocity_solid.load(counter)
        displacement_solid.new_average = displacement_solid.load(counter)

        velocity_fluid.save(velocity_fluid.new)
        displacement_fluid.save(displacement_fluid.new)
        pressure_fluid.save(pressure_fluid.new)
        velocity_solid.save(velocity_solid.new)
        displacement_solid.save(displacement_solid.new)

    # Create time loop
    size = fluid_timeline.size - param.PARTIAL_LOAD
    if param.PARTIAL_LOAD == 0:
        first_time_step = True
    else:
        first_time_step = False
    counter = param.PARTIAL_LOAD
    for n in range(size):

        if adjoint:

            print(f"Current macro time-step {size - counter}")

        else:

            print(f"Current macro time-step {counter + 1}")

        # Perform decoupling
        decoupling(
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

        # Perform final iteration and save solutions
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
            save=True,
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
            save=True,
        )
        first_time_step = False

        # Update solution
        velocity_fluid.old_old.assign(velocity_fluid.old)
        displacement_fluid.old_old.assign(displacement_fluid.old)
        pressure_fluid.old_old.assign(pressure_fluid.old)
        velocity_solid.old_old.assign(velocity_solid.old)
        displacement_solid.old_old.assign(displacement_solid.old)
        velocity_fluid.old.assign(velocity_fluid.new)
        displacement_fluid.old.assign(displacement_fluid.new)
        pressure_fluid.old.assign(pressure_fluid.new)
        velocity_solid.old.assign(velocity_solid.new)
        displacement_solid.old.assign(displacement_solid.new)

        velocity_fluid.old_old_average.assign(velocity_fluid.old_average)
        displacement_fluid.old_old_average.assign(
            displacement_fluid.old_average
        )
        pressure_fluid.old_old_average.assign(pressure_fluid.old_average)
        velocity_solid.old_old_average.assign(velocity_solid.old_average)
        displacement_solid.old_old_average.assign(
            displacement_solid.old_average
        )
        velocity_fluid.old_average.assign(velocity_fluid.new_average)
        displacement_fluid.old_average.assign(displacement_fluid.new_average)
        pressure_fluid.old_average.assign(pressure_fluid.new_average)
        velocity_solid.old_average.assign(velocity_solid.new_average)
        displacement_solid.old_average.assign(displacement_solid.new_average)

        # Advance timeline
        if adjoint:

            fluid_macrotimestep = fluid_macrotimestep.before
            solid_macrotimestep = solid_macrotimestep.before

        else:

            fluid_macrotimestep = fluid_macrotimestep.after
            solid_macrotimestep = solid_macrotimestep.after
        counter += 1

    # Save initial values for the adjoint problem
    if adjoint:

        velocity_fluid.save(Function(fluid.function_space_split[0]))
        displacement_fluid.save(Function(fluid.function_space_split[1]))
        pressure_fluid.save(Function(fluid.function_space_split[2]))
        velocity_solid.save(Function(solid.function_space_split[0]))
        displacement_solid.save(Function(solid.function_space_split[1]))

    # Check convergence
    failed = 0
    for i in range(len(velocity_fluid.iterations)):
        failed += min(0, velocity_fluid.iterations[i])
    if failed < 0:
        print("The decoupling method failed at some point")

    # Test fluid velocity on the interface
    velocity_fluid_to_solid = mirror_function(velocity_fluid.new, solid, fluid)
    print(assemble(dot(velocity_fluid.new, velocity_fluid.new) * fluid.ds(1)))
    print(
        assemble(
            dot(velocity_fluid_to_solid, velocity_fluid_to_solid) * solid.ds(1)
        )
    )

    # Test solid velocity on the interface
    velocity_solid_to_fluid = mirror_function(velocity_solid.new, fluid, solid)
    print(assemble(dot(velocity_solid.new, velocity_solid.new) * solid.ds(1)))
    print(
        assemble(
            dot(velocity_solid_to_fluid, velocity_solid_to_fluid) * fluid.ds(1)
        )
    )

    # Test fluid displacement on the interface
    displacement_fluid_to_solid = mirror_function(
        displacement_fluid.new, solid, fluid
    )
    print(
        assemble(
            dot(displacement_fluid.new, displacement_fluid.new) * fluid.ds(1)
        )
    )
    print(
        assemble(
            dot(displacement_fluid_to_solid, displacement_fluid_to_solid)
            * solid.ds(1)
        )
    )

    # Test solid displacement on the interface
    displacement_solid_to_fluid = mirror_function(
        displacement_solid.new, fluid, solid
    )
    print(
        assemble(
            dot(displacement_solid.new, displacement_solid.new) * solid.ds(1)
        )
    )
    print(
        assemble(
            dot(displacement_solid_to_fluid, displacement_solid_to_fluid)
            * fluid.ds(1)
        )
    )

    # Test fluid stress
    fluid_test_function = TestFunction(fluid.function_space)
    (
        fluid_first_test_function,
        fluid_second_test_function,
        fluid_third_test_function,
    ) = split(fluid_test_function)
    solid_test_function = TestFunction(solid.function_space)
    (
        solid_first_test_function,
        solid_second_test_function,
    ) = split(solid_test_function)
    stress = Expression(("x[0]", "x[1]"), degree=1)
    fluid_form = dot(stress, fluid_first_test_function) * fluid.ds(1)
    fluid_form_assemble = assemble(fluid_form)
    fluid_form_vector = fluid_form_assemble.get_local()
    solid_form = dot(stress, solid_first_test_function) * solid.ds(1)
    solid_form_assemble = assemble(solid_form)
    solid_form_vector = solid_form_assemble.get_local()
    form_fluid_to_solid = mirror_form(fluid_form_vector, solid, fluid)
    tol = 1.0e-12
    counter = 0
    for index in range(len(solid_form_vector)):
        if abs(solid_form_vector[index] - form_fluid_to_solid[index]) >= tol:
            print(solid_form_vector[index])
            print(form_fluid_to_solid[index])
            counter += 1
    print(counter)
    print(len(solid_form_vector))

    fluid_stress = det(deformation_gradient(displacement_fluid.new)) * dot(
        sigma_fluid(
            velocity_fluid.new,
            displacement_fluid.new,
            pressure_fluid.new,
            fluid,
            param,
        )
        * inv(deformation_gradient(displacement_fluid.new)).T,
        fluid.normal_vector,
    )
    solid_stress = dot(
        sigma_solid(displacement_solid.new, solid, param), solid.normal_vector
    )
    print("stress")
    print(assemble(dot(fluid_stress, Constant((1.0, 1.0))) * fluid.ds(1)))
    print(assemble(dot(solid_stress, Constant((1.0, 1.0))) * solid.ds(1)))
