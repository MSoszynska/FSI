from fenics import (
    project,
    assemble,
    dot,
    grad,
    inner,
    Function,
    assign,
    Constant,
    File,
    near,
    DirichletBC,
    Constant,
    Point,
    action,
)
from forms import (
    form_fluid,
    form_solid,
    form_fluid_interface,
    form_solid_interface,
    form_fluid_adjoint,
    form_solid_adjoint,
    form_fluid_interface_adjoint,
    form_solid_interface_adjoint,
    characteristic_function_fluid,
    characteristic_function_solid,
    form_fluid_velocity_derivative,
    form_fluid_displacement_derivative,
    form_fluid_velocity_derivative_adjoint,
    form_fluid_displacement_derivative_adjoint,
    gauss,
    sigma_fluid,
    external_force,
)
from parameters import Parameters
from spaces import Space
from time_structure import MicroTimeStep, MacroTimeStep, TimeLine
from coupling import mirror_function

# Copy solutions
def copy_list_forward(
    space: Space,
    timeline: TimeLine,
    function_name,
    subspace_index,
    param: Parameters,
    adjoint,
    adjust_size,
    homogenize=False,
):

    array = []
    if param.PARTIAL_COMPUTE is not None:
        if adjust_size:
            if param.PARTIAL_COMPUTE[0] != 0:
                starting_point = param.PARTIAL_COMPUTE[0] - 2
                size = param.PARTIAL_COMPUTE[1] + 2
            else:
                starting_point = param.PARTIAL_COMPUTE[0]
                size = param.PARTIAL_COMPUTE[1]
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        macrotimestep_adjust = timeline.head
        for i in range(starting_point):
            macrotimestep_adjust = macrotimestep_adjust.after
        macrotimestep = macrotimestep_adjust
        global_size = size
    else:
        macrotimestep = timeline.head
        global_size = timeline.size
    for n in range(global_size):

        microtimestep = macrotimestep.head
        local_size = macrotimestep.size
        for m in range(local_size):

            function = microtimestep.functions[function_name]
            if not microtimestep.after is None:

                if homogenize:
                    function_homogenize = Function(space.function_space)
                    assign(function_homogenize.sub(subspace_index), function)
                    for boundary in space.boundaries(
                        microtimestep.point, param, adjoint
                    ):
                        boundary.homogenize()
                        boundary.apply(function_homogenize.vector())
                    if space.name == "fluid":
                        (
                            function_homogenize_first,
                            function_homogenize_second,
                            function_homogenize_third,
                        ) = function_homogenize.split()
                    else:
                        (
                            function_homogenize_first,
                            function_homogenize_second,
                        ) = function_homogenize.split()
                    if subspace_index == 0:
                        array.append(
                            function_homogenize_first.copy(deepcopy=True)
                        )
                    if subspace_index == 1:
                        array.append(
                            function_homogenize_second.copy(deepcopy=True)
                        )
                    if subspace_index == 2:
                        array.append(
                            function_homogenize_third.copy(deepcopy=True)
                        )
                else:
                    array.append(function.copy(deepcopy=True))

            if (microtimestep.after is None) and (n == global_size - 1):

                if homogenize:
                    function_homogenize = Function(space.function_space)
                    assign(function_homogenize.sub(subspace_index), function)
                    for boundary in space.boundaries(
                        microtimestep.point, param, adjoint
                    ):
                        boundary.homogenize()
                        boundary.apply(function_homogenize.vector())
                    if space.name == "fluid":
                        (
                            function_homogenize_first,
                            function_homogenize_second,
                            function_homogenize_third,
                        ) = function_homogenize.split()
                    else:
                        (
                            function_homogenize_first,
                            function_homogenize_second,
                        ) = function_homogenize.split()
                    if subspace_index == 0:
                        array.append(
                            function_homogenize_first.copy(deepcopy=True)
                        )
                    if subspace_index == 1:
                        array.append(
                            function_homogenize_second.copy(deepcopy=True)
                        )
                    if subspace_index == 2:
                        array.append(
                            function_homogenize_third.copy(deepcopy=True)
                        )
                else:
                    array.append(function.copy(deepcopy=True))

            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return array


# Extrapolate solutions
def extrapolate_list_forward(
    space: Space,
    space_timeline: TimeLine,
    space_interface: Space,
    space_interface_timeline: TimeLine,
    function_name,
    subspace_index,
    param: Parameters,
    adjust_size,
):
    if param.PARTIAL_COMPUTE is not None:
        if adjust_size:
            if param.PARTIAL_COMPUTE[0] != 0:
                starting_point = param.PARTIAL_COMPUTE[0] - 2
                size = param.PARTIAL_COMPUTE[1] + 2
            else:
                starting_point = param.PARTIAL_COMPUTE[0]
                size = param.PARTIAL_COMPUTE[1]
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        space_macrotimestep_adjust = space_timeline.head
        space_interface_macrotimestep_adjust = space_interface_timeline.head
        for i in range(starting_point):
            space_macrotimestep_adjust = space_macrotimestep_adjust.after
            space_interface_macrotimestep_adjust = (
                space_interface_macrotimestep_adjust.after
            )
        space_macrotimestep = space_macrotimestep_adjust
        space_interface_macrotimestep = space_interface_macrotimestep_adjust
        global_size = size
    else:
        space_macrotimestep = space_timeline.head
        space_interface_macrotimestep = space_interface_timeline.head
        global_size = space_timeline.size
    array = []
    function = space_interface_macrotimestep.head.functions[function_name]
    array.append(function)
    for n in range(global_size):

        space_microtimestep = space_macrotimestep.head
        local_size = space_macrotimestep.size - 1
        for m in range(local_size):

            extrapolation_proportion = (
                space_macrotimestep.tail.point
                - space_microtimestep.after.point
            ) / space_macrotimestep.dt
            function_old = space_interface_macrotimestep.head.functions[
                function_name
            ]
            function_new = space_interface_macrotimestep.tail.functions[
                function_name
            ]
            array.append(
                project(
                    extrapolation_proportion * function_old
                    + (1.0 - extrapolation_proportion) * function_new,
                    space_interface.function_space_split[subspace_index],
                )
            )
            space_microtimestep = space_microtimestep.after

        space_macrotimestep = space_macrotimestep.after
        space_interface_macrotimestep = space_interface_macrotimestep.after

    return array


# Define linear extrapolation
def linear_extrapolation(array, m, time, microtimestep: MicroTimeStep):

    time_step_size = microtimestep.before.dt
    point = microtimestep.point

    return (array[m] - array[m - 1]) / time_step_size * time + (
        array[m - 1] * point - array[m] * (point - time_step_size)
    ) / time_step_size


# Define reconstruction of the primal problem
def primal_reconstruction(array, m, time, microtimestep: MicroTimeStep):

    time_step_size = microtimestep.before.dt
    point = microtimestep.point
    a = (array[m + 1] - 2 * array[m] + array[m - 1]) / (
        2.0 * time_step_size * time_step_size
    )
    b = (
        (time_step_size - 2.0 * point) * array[m + 1]
        + 4 * point * array[m]
        + (-time_step_size - 2.0 * point) * array[m - 1]
    ) / (2.0 * time_step_size * time_step_size)
    c = (
        (-time_step_size * point + point * point) * array[m + 1]
        + (2.0 * time_step_size * time_step_size - 2.0 * point * point)
        * array[m]
        + (time_step_size * point + point * point) * array[m - 1]
    ) / (2.0 * time_step_size * time_step_size)

    return a * time * time + b * time + c
    # return 0.0 * a


def primal_derivative(array, m, time, microtimestep: MicroTimeStep):

    time_step_size = microtimestep.before.dt
    point = microtimestep.point
    a = (array[m + 1] - 2 * array[m] + array[m - 1]) / (
        2.0 * time_step_size * time_step_size
    )
    b = (
        (time_step_size - 2.0 * point) * array[m + 1]
        + 4 * point * array[m]
        + (-time_step_size - 2.0 * point) * array[m - 1]
    ) / (2.0 * time_step_size * time_step_size)

    return 2.0 * a * time + b
    # return 0.0 * a


# Define reconstruction of the adjoint problem
def adjoint_reconstruction(array, m, time, microtimestep, macrotimestep):

    size = len(array) - 1
    if m == 1 or m == size:

        return array[m]
        # return 0.0 * array[m]

    else:

        if microtimestep.before.before is None:

            t_average_before = 0.5 * (
                microtimestep.before.point
                + macrotimestep.microtimestep_before.point
            )

        else:

            t_average_before = 0.5 * (
                microtimestep.before.point + microtimestep.before.before.point
            )

        if microtimestep.after is None:

            t_average_after = 0.5 * (
                microtimestep.point + macrotimestep.microtimestep_after.point
            )

        else:

            t_average_after = 0.5 * (
                microtimestep.point + microtimestep.after.point
            )

        return (time - t_average_before) / (
            t_average_after - t_average_before
        ) * array[m + 1] + (time - t_average_after) / (
            t_average_before - t_average_after
        ) * array[
            m - 1
        ]
        # return 0.0 * array[m - 1]


# Compute goal functionals
def goal_functional_fluid(
    fluid: Space,
    fluid_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_adjust = fluid_timeline.head
        for i in range(starting_point):
            macrotimestep_adjust = macrotimestep_adjust.after
        macrotimestep = macrotimestep_adjust
        global_size = size
    else:
        macrotimestep = fluid_timeline.head
        global_size = fluid_timeline.size

    # Prepare arrays of solutions
    velocity_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_velocity", 0, param, False, False
    )
    pressure_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_pressure", 1, param, False, False
    )

    goal_functional = []
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            result = 0.0
            time_step_size = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            result += (
                0.5
                * time_step_size
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep
                    ),
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep
                    ),
                )
                * fluid.dx
            )
            result += (
                0.5
                * time_step_size
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep
                    ),
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep
                    ),
                )
                * fluid.dx
            )
            goal_functional.append(assemble(result))
            m += 1
            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return goal_functional


def goal_functional_solid(
    solid,
    solid_timeline,
    param,
):

    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_adjust = solid_timeline.head
        for i in range(starting_point):
            macrotimestep_adjust = macrotimestep_adjust.after
        macrotimestep = macrotimestep_adjust
        global_size = size
    else:
        macrotimestep = solid_timeline.head
        global_size = solid_timeline.size

    # Prepare arrays of solutions
    velocity_solid_array = copy_list_forward(
        solid, solid_timeline, "primal_velocity", 0, param, False, False
    )
    displacement_solid_array = copy_list_forward(
        solid, solid_timeline, "primal_displacement", 1, param, False, False
    )
    goal_functional = []
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            result = 0.0
            time_step_size = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            result += (
                0.5
                * time_step_size
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    linear_extrapolation(
                        displacement_solid_array, m, gauss_1, microtimestep
                    ),
                    linear_extrapolation(
                        displacement_solid_array, m, gauss_1, microtimestep
                    ),
                )
                * solid.dx
            )
            result += (
                0.5
                * time_step_size
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    linear_extrapolation(
                        displacement_solid_array, m, gauss_2, microtimestep
                    ),
                    linear_extrapolation(
                        displacement_solid_array, m, gauss_2, microtimestep
                    ),
                )
                * solid.dx
            )
            goal_functional.append(assemble(result))
            m += 1
            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return goal_functional


# Compute primal residual of the fluid subproblem
def primal_residual_fluid(
    fluid: Space,
    solid: Space,
    fluid_timeline: TimeLine,
    solid_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            starting_point = param.PARTIAL_COMPUTE[0] - 2
            size = param.PARTIAL_COMPUTE[1] + 2
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        macrotimestep_fluid_adjust = fluid_timeline.head
        for i in range(starting_point):
            macrotimestep_fluid_adjust = macrotimestep_fluid_adjust.after
        macrotimestep_fluid = macrotimestep_fluid_adjust
        global_size = size
    else:
        macrotimestep_fluid = fluid_timeline.head
        global_size = fluid_timeline.size

    # Prepare arrays of solutions
    velocity_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_velocity", 0, param, False, True
    )
    displacement_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_displacement", 1, param, False, True
    )
    pressure_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_pressure", 2, param, False, True
    )
    velocity_solid_array = extrapolate_list_forward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "primal_velocity",
        0,
        param,
        True,
    )
    displacement_solid_array = extrapolate_list_forward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "primal_displacement",
        1,
        param,
        True,
    )
    velocity_fluid_adjoint_array = copy_list_forward(
        fluid, fluid_timeline, "adjoint_velocity", 0, param, True, True, True
    )
    displacement_fluid_adjoint_array = copy_list_forward(
        fluid,
        fluid_timeline,
        "adjoint_displacement",
        1,
        param,
        True,
        True,
        True,
    )
    pressure_fluid_adjoint_array = copy_list_forward(
        fluid, fluid_timeline, "adjoint_pressure", 2, param, True, True, True
    )
    residuals = []
    m = 1
    for n in range(global_size):

        microtimestep_fluid_before = macrotimestep_fluid.head
        microtimestep_fluid = macrotimestep_fluid.head.after
        if microtimestep_fluid.after is None:
            if macrotimestep_fluid.after is not None:
                microtimestep_fluid_after = (
                    macrotimestep_fluid.after.head.after
                )
            else:
                microtimestep_fluid_after = None
        else:
            microtimestep_fluid_after = microtimestep_fluid.after
        time = microtimestep_fluid.point
        local_size = macrotimestep_fluid.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_fluid.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_fluid)

            lhs += (
                0.5
                * time_step_size
                * form_fluid_velocity_derivative(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    param,
                )
                * dot(
                    (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_velocity_derivative(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    param,
                )
                * dot(
                    (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs -= (
                0.5
                * time_step_size
                * dot(
                    dot(
                        form_fluid_displacement_derivative(
                            linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            param,
                        ),
                        (
                            displacement_fluid_array[m]
                            - displacement_fluid_array[m - 1]
                        )
                        / time_step_size,
                    ),
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs -= (
                0.5
                * time_step_size
                * dot(
                    dot(
                        form_fluid_displacement_derivative(
                            linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            param,
                        ),
                        (
                            displacement_fluid_array[m]
                            - displacement_fluid_array[m - 1]
                        )
                        / time_step_size,
                    ),
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - displacement_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        pressure_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - pressure_fluid_adjoint_array[m],
                    fluid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - displacement_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        pressure_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - pressure_fluid_adjoint_array[m],
                    fluid,
                    param,
                )
            )
            lhs -= (
                0.5
                * time_step_size
                * form_fluid_interface(
                    mirror_function(
                        linear_extrapolation(
                            velocity_solid_array,
                            m,
                            gauss_1,
                            microtimestep_fluid,
                        ),
                        fluid,
                        solid,
                    ),
                    mirror_function(
                        linear_extrapolation(
                            displacement_solid_array,
                            m,
                            gauss_1,
                            microtimestep_fluid,
                        ),
                        fluid,
                        solid,
                    ),
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - displacement_fluid_adjoint_array[m],
                    fluid,
                    param,
                )
            )
            lhs -= (
                0.5
                * time_step_size
                * form_fluid_interface(
                    mirror_function(
                        linear_extrapolation(
                            velocity_solid_array,
                            m,
                            gauss_2,
                            microtimestep_fluid,
                        ),
                        fluid,
                        solid,
                    ),
                    mirror_function(
                        linear_extrapolation(
                            displacement_solid_array,
                            m,
                            gauss_2,
                            microtimestep_fluid,
                        ),
                        fluid,
                        solid,
                    ),
                    adjoint_reconstruction(
                        velocity_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - velocity_fluid_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_fluid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                        macrotimestep_fluid,
                    )
                    - displacement_fluid_adjoint_array[m],
                    fluid,
                    param,
                )
            )
            rhs += (
                0.5
                * param.RHO_FLUID
                * time_step_size
                * characteristic_function_fluid(param)
                * dot(
                    external_force(gauss_1, param),
                    (
                        adjoint_reconstruction(
                            velocity_fluid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep_fluid,
                            macrotimestep_fluid,
                        )
                        - velocity_fluid_adjoint_array[m]
                    ),
                )
                * fluid.dx
            )
            rhs += (
                0.5
                * param.RHO_FLUID
                * time_step_size
                * characteristic_function_fluid(param)
                * dot(
                    external_force(gauss_2, param),
                    (
                        adjoint_reconstruction(
                            velocity_fluid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep_fluid,
                            macrotimestep_fluid,
                        )
                        - velocity_fluid_adjoint_array[m]
                    ),
                )
                * fluid.dx
            )

            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))
            m += 1
            microtimestep_fluid_before = microtimestep_fluid_before.after
            microtimestep_fluid = microtimestep_fluid.after

        macrotimestep_fluid = macrotimestep_fluid.after

    return residuals


# Compute primal residual of the solid subproblem
def primal_residual_solid(
    solid: Space,
    fluid: Space,
    solid_timeline: TimeLine,
    fluid_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            starting_point = param.PARTIAL_COMPUTE[0] - 2
            size = param.PARTIAL_COMPUTE[1] + 2
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        macrotimestep_solid_adjust = solid_timeline.head
        for i in range(starting_point):
            macrotimestep_solid_adjust = macrotimestep_solid_adjust.after
        macrotimestep_solid = macrotimestep_solid_adjust
        global_size = size
    else:
        macrotimestep_solid = solid_timeline.head
        global_size = solid_timeline.size

    # Prepare arrays of solutions
    velocity_solid_array = copy_list_forward(
        solid, solid_timeline, "primal_velocity", 0, param, False, True
    )
    displacement_solid_array = copy_list_forward(
        solid, solid_timeline, "primal_displacement", 1, param, False, True
    )
    velocity_solid_array_homogenize = copy_list_forward(
        solid, solid_timeline, "primal_velocity", 0, param, False, True, True
    )
    displacement_solid_array_homogenize = copy_list_forward(
        solid,
        solid_timeline,
        "primal_displacement",
        1,
        param,
        False,
        True,
        True,
    )
    velocity_fluid_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "primal_velocity",
        0,
        param,
        True,
    )
    displacement_fluid_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "primal_displacement",
        1,
        param,
        True,
    )
    pressure_fluid_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "primal_pressure",
        2,
        param,
        True,
    )
    velocity_solid_adjoint_array = copy_list_forward(
        solid, solid_timeline, "adjoint_velocity", 0, param, True, True, True
    )
    displacement_solid_adjoint_array = copy_list_forward(
        solid,
        solid_timeline,
        "adjoint_displacement",
        1,
        param,
        True,
        True,
        True,
    )
    residuals = []
    m = 1
    for n in range(global_size):

        microtimestep_solid_before = macrotimestep_solid.head
        microtimestep_solid = macrotimestep_solid.head.after
        if microtimestep_solid.after is None:
            if macrotimestep_solid.after is not None:
                microtimestep_solid_after = (
                    macrotimestep_solid.after.head.after
                )
            else:
                microtimestep_solid_after = None
        else:
            microtimestep_solid_after = microtimestep_solid.after
        time = microtimestep_solid.point
        local_size = macrotimestep_solid.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            lhs_interface = 0.0
            time_step_size = microtimestep_solid.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_solid)

            lhs += (
                0.5
                * param.RHO_SOLID
                * time_step_size
                * dot(
                    (velocity_solid_array[m] - velocity_solid_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        velocity_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - velocity_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * param.RHO_SOLID
                * time_step_size
                * dot(
                    (velocity_solid_array[m] - velocity_solid_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        velocity_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - velocity_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    (
                        displacement_solid_array[m]
                        - displacement_solid_array[m - 1]
                    )
                    / time_step_size,
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - displacement_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    (
                        displacement_solid_array[m]
                        - displacement_solid_array[m - 1]
                    )
                    / time_step_size,
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - displacement_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_solid(
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_1, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    adjoint_reconstruction(
                        velocity_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - velocity_solid_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - displacement_solid_adjoint_array[m],
                    solid,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_solid(
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_2, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    adjoint_reconstruction(
                        velocity_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - velocity_solid_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_solid_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                        macrotimestep_solid,
                    )
                    - displacement_solid_adjoint_array[m],
                    solid,
                    param,
                )
            )
            rhs += (
                0.5
                * param.RHO_SOLID
                * time_step_size
                * characteristic_function_solid(param)
                * dot(
                    external_force(gauss_1, param),
                    (
                        adjoint_reconstruction(
                            velocity_solid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep_solid,
                            macrotimestep_solid,
                        )
                        - velocity_solid_adjoint_array[m]
                    ),
                )
                * solid.dx
            )
            rhs += (
                0.5
                * param.RHO_SOLID
                * time_step_size
                * characteristic_function_solid(param)
                * dot(
                    external_force(gauss_2, param),
                    (
                        adjoint_reconstruction(
                            velocity_solid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep_solid,
                            macrotimestep_solid,
                        )
                        - velocity_solid_adjoint_array[m]
                    ),
                )
                * solid.dx
            )
            lhs_interface += (
                0.5
                * time_step_size
                * form_solid_interface(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_1, microtimestep_solid
                    ),
                    mirror_function(
                        adjoint_reconstruction(
                            velocity_solid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep_solid,
                            macrotimestep_solid,
                        )
                        - velocity_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    mirror_function(
                        adjoint_reconstruction(
                            displacement_solid_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep_solid,
                            macrotimestep_solid,
                        )
                        - displacement_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    fluid,
                    param,
                )
            )
            lhs_interface += (
                0.5
                * time_step_size
                * form_solid_interface(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_2, microtimestep_solid
                    ),
                    mirror_function(
                        adjoint_reconstruction(
                            velocity_solid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep_solid,
                            macrotimestep_solid,
                        )
                        - velocity_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    mirror_function(
                        adjoint_reconstruction(
                            displacement_solid_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep_solid,
                            macrotimestep_solid,
                        )
                        - displacement_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    fluid,
                    param,
                )
            )

            residuals.append(
                assemble(0.5 * rhs - 0.5 * lhs)
                + assemble(-0.5 * lhs_interface)
            )
            m += 1
            microtimestep_solid_before = microtimestep_solid_before.after
            microtimestep_solid = microtimestep_solid.after

        macrotimestep_solid = macrotimestep_solid.after

    return residuals


# Compute adjoint residual of the fluid subproblem
def adjoint_residual_fluid(
    fluid: Space,
    solid: Space,
    fluid_timeline: TimeLine,
    solid_timeline: TimeLine,
    param: Parameters,
):
    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_fluid_adjust = fluid_timeline.head
        for i in range(starting_point):
            macrotimestep_fluid_adjust = macrotimestep_fluid_adjust.after
        macrotimestep_fluid = macrotimestep_fluid_adjust
        global_size = size
    else:
        macrotimestep_fluid = fluid_timeline.head
        global_size = fluid_timeline.size

    # Prepare arrays of solutions
    velocity_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_velocity", 0, param, False, False, True
    )
    displacement_fluid_array = copy_list_forward(
        fluid,
        fluid_timeline,
        "primal_displacement",
        1,
        param,
        False,
        False,
        True,
    )
    pressure_fluid_array = copy_list_forward(
        fluid, fluid_timeline, "primal_pressure", 2, param, False, False, True
    )
    velocity_fluid_adjoint_array = copy_list_forward(
        fluid, fluid_timeline, "adjoint_velocity", 0, param, True, False
    )
    displacement_fluid_adjoint_array = copy_list_forward(
        fluid, fluid_timeline, "adjoint_displacement", 1, param, True, False
    )
    pressure_fluid_adjoint_array = copy_list_forward(
        fluid, fluid_timeline, "adjoint_pressure", 1, param, True, False
    )
    velocity_solid_adjoint_array = extrapolate_list_forward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "adjoint_velocity",
        0,
        param,
        False,
    )
    displacement_solid_adjoint_array = extrapolate_list_forward(
        fluid,
        fluid_timeline,
        solid,
        solid_timeline,
        "adjoint_displacement",
        1,
        param,
        False,
    )
    residuals = []
    left = True
    m = 1
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)
    for n in range(global_size):

        microtimestep_fluid_before = macrotimestep_fluid.head
        microtimestep_fluid = macrotimestep_fluid.head.after
        local_size = macrotimestep_fluid.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_fluid.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_fluid)
            if left:

                l = m
                microtimestep_fluid_adjust = microtimestep_fluid
                left = False

            else:

                l = m - 1
                if microtimestep_fluid.before.before is None:

                    microtimestep_fluid_adjust = (
                        macrotimestep_fluid.microtimestep_before.after
                    )

                else:

                    microtimestep_fluid_adjust = microtimestep_fluid.before
                left = True

            lhs += (
                0.5
                * time_step_size
                * form_fluid_velocity_derivative_adjoint(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    fluid,
                    param,
                    error_estimate=True,
                )
                * dot(
                    (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step_size,
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_velocity_derivative_adjoint(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    fluid,
                    param,
                    error_estimate=True,
                )
                * dot(
                    (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step_size,
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_velocity_derivative(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    param,
                )
                * dot(
                    primal_derivative(
                        velocity_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step_size,
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_velocity_derivative(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    param,
                )
                * dot(
                    primal_derivative(
                        velocity_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - (velocity_fluid_array[m] - velocity_fluid_array[m - 1])
                    / time_step_size,
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs -= (
                0.5
                * time_step_size
                * dot(
                    dot(
                        form_fluid_displacement_derivative_adjoint(
                            linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            primal_reconstruction(
                                velocity_fluid_array,
                                l,
                                gauss_1,
                                microtimestep_fluid_adjust,
                            )
                            - linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            primal_reconstruction(
                                displacement_fluid_array,
                                l,
                                gauss_1,
                                microtimestep_fluid_adjust,
                            )
                            - linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            fluid,
                            param,
                            error_estimate=True,
                        ),
                        (
                            displacement_fluid_array[m]
                            - displacement_fluid_array[m - 1]
                        )
                        / time_step_size,
                    ),
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs -= (
                0.5
                * time_step_size
                * dot(
                    dot(
                        form_fluid_displacement_derivative_adjoint(
                            linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            primal_reconstruction(
                                velocity_fluid_array,
                                l,
                                gauss_2,
                                microtimestep_fluid_adjust,
                            )
                            - linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            primal_reconstruction(
                                displacement_fluid_array,
                                l,
                                gauss_2,
                                microtimestep_fluid_adjust,
                            )
                            - linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            fluid,
                            param,
                            error_estimate=True,
                        ),
                        (
                            displacement_fluid_array[m]
                            - displacement_fluid_array[m - 1]
                        )
                        / time_step_size,
                    ),
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs -= (
                0.5
                * time_step_size
                * dot(
                    dot(
                        form_fluid_displacement_derivative(
                            linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_1,
                                microtimestep_fluid,
                            ),
                            param,
                        ),
                        primal_derivative(
                            displacement_fluid_array,
                            l,
                            gauss_1,
                            microtimestep_fluid_adjust,
                        )
                        - (
                            displacement_fluid_array[m]
                            - displacement_fluid_array[m - 1]
                        )
                        / time_step_size,
                    ),
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs -= (
                0.5
                * time_step_size
                * dot(
                    dot(
                        form_fluid_displacement_derivative(
                            linear_extrapolation(
                                velocity_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            linear_extrapolation(
                                displacement_fluid_array,
                                m,
                                gauss_2,
                                microtimestep_fluid,
                            ),
                            param,
                        ),
                        primal_derivative(
                            displacement_fluid_array,
                            l,
                            gauss_2,
                            microtimestep_fluid_adjust,
                        )
                        - (
                            displacement_fluid_array[m]
                            - displacement_fluid_array[m - 1]
                        )
                        / time_step_size,
                    ),
                    velocity_fluid_adjoint_array[m],
                )
                * fluid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_adjoint(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    velocity_fluid_adjoint_array[m],
                    displacement_fluid_adjoint_array[m],
                    pressure_fluid_adjoint_array[m],
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    primal_reconstruction(
                        pressure_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        pressure_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    fluid,
                    param,
                    error_estimate=True,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_adjoint(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    velocity_fluid_adjoint_array[m],
                    displacement_fluid_adjoint_array[m],
                    pressure_fluid_adjoint_array[m],
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    primal_reconstruction(
                        pressure_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        pressure_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    fluid,
                    param,
                    error_estimate=True,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_interface_adjoint(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    mirror_function(
                        velocity_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    mirror_function(
                        displacement_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_1,
                        microtimestep_fluid,
                    ),
                    primal_reconstruction(
                        pressure_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        pressure_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    fluid,
                    param,
                    error_estimate=True,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_fluid_interface_adjoint(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    linear_extrapolation(
                        pressure_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    mirror_function(
                        velocity_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    mirror_function(
                        displacement_solid_adjoint_array[m],
                        fluid,
                        solid,
                    ),
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        displacement_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_fluid_array,
                        m,
                        gauss_2,
                        microtimestep_fluid,
                    ),
                    primal_reconstruction(
                        pressure_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        pressure_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    fluid,
                    param,
                    error_estimate=True,
                )
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_1,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_1, microtimestep_fluid
                    ),
                )
                * fluid.dx
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * param.NU
                * characteristic_function_fluid(param)
                * dot(
                    linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                    primal_reconstruction(
                        velocity_fluid_array,
                        l,
                        gauss_2,
                        microtimestep_fluid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_fluid_array, m, gauss_2, microtimestep_fluid
                    ),
                )
                * fluid.dx
            )

            residuals.append(assemble(rhs - 0.5 * lhs))
            m += 1
            microtimestep_fluid_before = microtimestep_fluid_before.after
            microtimestep_fluid = microtimestep_fluid.after

        macrotimestep_fluid = macrotimestep_fluid.after
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)

    return residuals


# Compute adjoint residual of the solid subproblem
def adjoint_residual_solid(
    solid: Space,
    fluid: Space,
    solid_timeline: TimeLine,
    fluid_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_solid_adjust = solid_timeline.head
        for i in range(starting_point):
            macrotimestep_solid_adjust = macrotimestep_solid_adjust.after
        macrotimestep_solid = macrotimestep_solid_adjust
        global_size = size
    else:
        macrotimestep_solid = solid_timeline.head
        global_size = solid_timeline.size

    # Prepare arrays of solutions
    velocity_solid_array = copy_list_forward(
        solid, solid_timeline, "primal_velocity", 0, param, False, False, True
    )
    displacement_solid_array = copy_list_forward(
        solid,
        solid_timeline,
        "primal_displacement",
        1,
        param,
        False,
        False,
        True,
    )
    velocity_solid_adjoint_array = copy_list_forward(
        solid, solid_timeline, "adjoint_velocity", 0, param, True, False
    )
    displacement_solid_adjoint_array = copy_list_forward(
        solid, solid_timeline, "adjoint_displacement", 1, param, True, False
    )
    velocity_fluid_adjoint_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "adjoint_velocity",
        0,
        param,
        False,
    )
    displacement_fluid_adjoint_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "adjoint_displacement",
        1,
        param,
        False,
    )
    pressure_fluid_adjoint_array = extrapolate_list_forward(
        solid,
        solid_timeline,
        fluid,
        fluid_timeline,
        "adjoint_pressure",
        2,
        param,
        False,
    )
    residuals = []
    left = True
    m = 1
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)
    for n in range(global_size):

        microtimestep_solid_before = macrotimestep_solid.head
        microtimestep_solid = macrotimestep_solid.head.after
        local_size = macrotimestep_solid.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_solid.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_solid)
            if left:

                l = m
                microtimestep_solid_adjust = microtimestep_solid
                left = False

            else:

                l = m - 1
                if microtimestep_solid.before.before is None:

                    microtimestep_solid_adjust = (
                        macrotimestep_solid.microtimestep_before.after
                    )

                else:

                    microtimestep_solid_adjust = microtimestep_solid.before
                left = True

            lhs += (
                0.5
                * param.RHO_SOLID
                * time_step_size
                * dot(
                    primal_derivative(
                        velocity_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - (velocity_solid_array[m] - velocity_solid_array[m - 1])
                    / time_step_size,
                    velocity_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * param.RHO_SOLID
                * time_step_size
                * dot(
                    primal_derivative(
                        velocity_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - (velocity_solid_array[m] - velocity_solid_array[m - 1])
                    / time_step_size,
                    velocity_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    primal_derivative(
                        displacement_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - (
                        displacement_solid_array[m]
                        - displacement_solid_array[m - 1]
                    )
                    / time_step_size,
                    displacement_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    primal_derivative(
                        displacement_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - (
                        displacement_solid_array[m]
                        - displacement_solid_array[m - 1]
                    )
                    / time_step_size,
                    displacement_solid_adjoint_array[m],
                )
                * solid.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_solid_adjoint(
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_1, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    velocity_solid_adjoint_array[m],
                    displacement_solid_adjoint_array[m],
                    primal_reconstruction(
                        velocity_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_solid_array, m, gauss_1, microtimestep_solid
                    ),
                    primal_reconstruction(
                        displacement_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    solid,
                    param,
                    error_estimate=True,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_solid_adjoint(
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_2, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    velocity_solid_adjoint_array[m],
                    displacement_solid_adjoint_array[m],
                    primal_reconstruction(
                        velocity_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_solid_array, m, gauss_2, microtimestep_solid
                    ),
                    primal_reconstruction(
                        displacement_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    solid,
                    param,
                    error_estimate=True,
                )
            )
            lhs -= (
                0.5
                * time_step_size
                * form_solid_interface_adjoint(
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_1, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    mirror_function(
                        velocity_fluid_adjoint_array[m],
                        solid,
                        fluid,
                    ),
                    mirror_function(
                        displacement_fluid_adjoint_array[m],
                        solid,
                        fluid,
                    ),
                    primal_reconstruction(
                        velocity_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_solid_array, m, gauss_1, microtimestep_solid
                    ),
                    primal_reconstruction(
                        displacement_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    solid,
                    param,
                    error_estimate=True,
                )
            )
            lhs -= (
                0.5
                * time_step_size
                * form_solid_interface_adjoint(
                    linear_extrapolation(
                        velocity_solid_array, m, gauss_2, microtimestep_solid
                    ),
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    mirror_function(
                        velocity_fluid_adjoint_array[m],
                        solid,
                        fluid,
                    ),
                    mirror_function(
                        displacement_fluid_adjoint_array[m],
                        solid,
                        fluid,
                    ),
                    primal_reconstruction(
                        velocity_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        velocity_solid_array, m, gauss_2, microtimestep_solid
                    ),
                    primal_reconstruction(
                        displacement_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    solid,
                    param,
                    error_estimate=True,
                )
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                    primal_reconstruction(
                        displacement_solid_array,
                        l,
                        gauss_1,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_1,
                        microtimestep_solid,
                    ),
                )
                * solid.dx
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * param.ZETA
                * characteristic_function_solid(param)
                * dot(
                    linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                    primal_reconstruction(
                        displacement_solid_array,
                        l,
                        gauss_2,
                        microtimestep_solid_adjust,
                    )
                    - linear_extrapolation(
                        displacement_solid_array,
                        m,
                        gauss_2,
                        microtimestep_solid,
                    ),
                )
                * solid.dx
            )

            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))
            m += 1
            microtimestep_solid_before = microtimestep_solid_before.after
            microtimestep_solid = microtimestep_solid.after

        macrotimestep_solid = macrotimestep_solid.after
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)

    return residuals
