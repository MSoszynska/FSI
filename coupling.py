import numpy as np
from fenics import (
    Function,
    FunctionSpace,
    vertex_to_dof_map,
    Expression,
    project,
    inner,
    Constant,
    DirichletBC,
    TrialFunction,
    TestFunction,
    split,
    dot,
    dx,
    ds,
    assemble,
    assign,
    TestFunction,
)
from spaces import Space, boundary

# Define unit vectors
def unit_vectors(space_dimension):
    if space_dimension == 1:
        return [Expression("1.0", degree=0)]
    elif space_dimension == 2:
        return [
            Expression(("1", "0"), degree=0),
            Expression(("0", "1"), degree=0),
        ]
    elif space_dimension == 3:
        return [
            Expression(("1", "0", "0"), degree=0),
            Expression(("0", "1", "0"), degree=0),
            Expression(("0", "0", "1"), degree=0),
        ]


# Define interface coordinates
def interface_coordinates(space: Space):

    coordinates = np.ascontiguousarray(
        space.function_space.tabulate_dof_coordinates()
    )
    unique_coordinates = np.unique(
        coordinates.view([("", coordinates.dtype)] * coordinates.shape[1])
    )

    return unique_coordinates.view(coordinates.dtype).reshape(
        (unique_coordinates.shape[0], coordinates.shape[1])
    )


# Define a table of coordinates on the interface
def interface_coordinates_transfer_table(
    space: Space, interface_coordinates, coordinate_index, collapse=False
):

    result = []
    tol = 1.0e-6
    for i in range(len(interface_coordinates)):
        characteristic_function = ["0.0", "0.0", "0.0", "0.0", "0.0"]
        characteristic_function[coordinate_index] = (
            "point_first - tol <= x[0] && "
            "x[0] <= point_first + tol && "
            "point_second - tol <= x[1] && "
            "x[1] <= point_second + tol ? 1.0 : 0.0"
        )
        characteristic_velocity = Expression(
            (characteristic_function[0], characteristic_function[1]),
            point_first=interface_coordinates[i][0],
            point_second=interface_coordinates[i][1],
            tol=tol,
            degree=1,
        )
        one_velocity = Constant((1.0, 1.0))
        characteristic_displacement = Expression(
            (characteristic_function[2], characteristic_function[3]),
            point_first=interface_coordinates[i][0],
            point_second=interface_coordinates[i][1],
            tol=tol,
            degree=1,
        )
        one_displacement = Constant((1.0, 1.0))
        if not collapse:
            if space.name == "fluid":
                characteristic_pressure = Expression(
                    characteristic_function[4],
                    point_first=interface_coordinates[i][0],
                    point_second=interface_coordinates[i][1],
                    tol=tol,
                    degree=1,
                )
                one_pressure = Constant(1.0)
                boundaries = [
                    DirichletBC(
                        space.function_space.sub(0),
                        characteristic_velocity,
                        boundary,
                    ),
                    DirichletBC(
                        space.function_space.sub(1),
                        characteristic_displacement,
                        boundary,
                    ),
                    DirichletBC(
                        space.function_space.sub(2),
                        characteristic_pressure,
                        boundary,
                    ),
                ]
                test_function = TestFunction(space.function_space)
                (
                    first_test_function,
                    second_test_function,
                    third_test_function,
                ) = split(test_function)
                right_hand_side = (
                    dot(one_velocity, first_test_function) * ds
                    + dot(one_displacement, second_test_function) * ds
                    # + dot(one_pressure, third_test_function) * ds
                )
            else:
                boundaries = [
                    DirichletBC(
                        space.function_space.sub(0),
                        characteristic_velocity,
                        boundary,
                    ),
                    DirichletBC(
                        space.function_space.sub(1),
                        characteristic_displacement,
                        boundary,
                    ),
                ]
                test_function = TestFunction(space.function_space)
                (first_test_function, second_test_function) = split(
                    test_function
                )
                right_hand_side = (
                    dot(one_velocity, first_test_function) * ds
                    + dot(one_displacement, second_test_function) * ds
                )
        else:
            boundaries = [
                DirichletBC(
                    space.function_space_split[0],
                    characteristic_velocity,
                    boundary,
                )
            ]
            first_test_function = TestFunction(space.function_space_split[0])
            right_hand_side = dot(one_velocity, first_test_function) * ds
        right_hand_side_assemble = assemble(right_hand_side)
        [boundary.apply(right_hand_side_assemble) for boundary in boundaries]
        right_hand_side_vector = right_hand_side_assemble.get_local()
        right_hand_side_numpy = np.array(right_hand_side_vector)
        right_hand_side_nonzero = np.array(np.nonzero(right_hand_side_numpy))
        if len(right_hand_side_nonzero[0] > 0):
            result.append(right_hand_side_nonzero[0][0])

    return result


# Define a table of coordinates for the extension from the interface
def extension_interface_coordinates_transfer_table(
    space: Space, interface_coordinates
):

    result_first_coordinate = []
    result_second_coordinate = []
    tol = 1.0e-6
    interface_table = space.function_space_split[0].tabulate_dof_coordinates()
    for i in range(len(interface_coordinates)):
        point_coordinates = interface_coordinates[i]
        j = 0
        stop = False
        while not stop:
            point_function_space = interface_table[j]
            if (
                abs(point_coordinates[0] - point_function_space[0]) < tol
                and abs(point_coordinates[1] - point_function_space[1]) < tol
            ):
                result_first_coordinate.append(j)
                result_second_coordinate.append(j + 1)
                stop = True
            else:
                j += 2

    return [result_first_coordinate, result_second_coordinate]


# Define a transfer of a function across the interface
def mirror_function(
    input, space: Space, space_interface: Space, collapse=False
):

    if not collapse:
        input_whole_space = Function(space_interface.function_space)
        input_project = project(input, space_interface.function_space_split[0])
        assign(input_whole_space.sub(0), input_project)
        input_vector = input_whole_space.vector()
        output = Function(space.function_space)
        output_vector = output.vector()
    else:
        input_project = project(input, space_interface.function_space_split[0])
        input_vector = input_project.vector()
        output = Function(space.function_space_split[0])
        output_vector = output.vector()
    for coordinate_index in range(2):
        input_interface_table = space_interface.interface_table[
            coordinate_index
        ]
        if not collapse:
            output_interface_table = space.interface_table[coordinate_index]
        else:
            output_interface_table = space.interface_table_collapse[
                coordinate_index
            ]
        for i in range(len(space.interface_coordinates)):
            output_vector[output_interface_table[i]] = input_vector[
                input_interface_table[i]
            ]
    if not collapse:
        if space.name == "fluid":
            (first_function, second_function, third_function) = output.split(
                True
            )
        else:
            (first_function, second_function) = output.split(True)
    else:
        first_function = output

    return first_function


# Define a transfer of a form across the interface
def mirror_form(input, space: Space, space_interface: Space):
    test_function = TestFunction(space.function_space)
    if space.name == "fluid":
        (
            first_test_function,
            second_test_function,
            third_test_function,
        ) = split(test_function)
        output = assemble(
            dot(Constant((0.0, 0.0)), first_test_function) * space.dx
            + dot(Constant((0.0, 0.0)), second_test_function) * space.dx
            + dot(Constant(0.0), third_test_function) * space.dx
        )
    else:
        (first_test_function, second_test_function) = split(test_function)
        output = assemble(
            dot(Constant((0.0, 0.0)), first_test_function) * space.dx
            + dot(Constant((0.0, 0.0)), second_test_function) * space.dx
        )
    for coordinate_index in range(2):
        input_interface_table = space_interface.interface_table[
            coordinate_index
        ]
        output_interface_table = space.interface_table[coordinate_index]
        for i in range(len(space.interface_coordinates)):
            output[output_interface_table[i]] = input[input_interface_table[i]]

    return output
