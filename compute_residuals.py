from error_estimate import (
    primal_residual_fluid,
    primal_residual_solid,
    adjoint_residual_fluid,
    adjoint_residual_solid,
    goal_functional_fluid,
    goal_functional_solid,
)
from parameters import Parameters
from spaces import Space
from time_structure import TimeLine


def compute_residuals(
    fluid: Space,
    solid: Space,
    param: Parameters,
    fluid_timeline: TimeLine,
    solid_timeline: TimeLine,
):

    # Create text file
    residuals_txt = open("residuals.txt", "a")

    # Compute residuals
    primal_fluid = primal_residual_fluid(
        fluid,
        solid,
        fluid_timeline,
        solid_timeline,
        param,
    )
    print(f"Primal residual for the fluid subproblem: " f"{sum(primal_fluid)}")
    residuals_txt.write(
        f"Primal residual for the fluid subproblem: "
        f"{sum(primal_fluid)} \r\n"
    )
    primal_solid = primal_residual_solid(
        solid,
        fluid,
        solid_timeline,
        fluid_timeline,
        param,
    )
    print(f"Primal residual for the solid subproblem: " f"{sum(primal_solid)}")
    residuals_txt.write(
        f"Primal residual for the solid subproblem: "
        f"{sum(primal_solid)} \r\n"
    )
    adjoint_fluid = adjoint_residual_fluid(
        fluid,
        solid,
        fluid_timeline,
        solid_timeline,
        param,
    )
    print(
        f"Adjoint residual for the fluid subproblem: " f"{sum(adjoint_fluid)}"
    )
    residuals_txt.write(
        f"Adjoint residual for the fluid subproblem: "
        f"{sum(adjoint_fluid)} \r\n"
    )
    adjoint_solid = adjoint_residual_solid(
        solid,
        fluid,
        solid_timeline,
        fluid_timeline,
        param,
    )
    print(
        f"Adjoint residual for the solid subproblem: " f"{sum(adjoint_solid)}"
    )
    residuals_txt.write(
        f"Adjoint residual for the solid subproblem: "
        f"{sum(adjoint_solid)} \r\n"
    )

    # Compute goal functional
    if param.GOAL_FUNCTIONAL_FLUID:

        goal_functional = goal_functional_fluid(
            fluid,
            fluid_timeline,
            param,
        )

    else:

        goal_functional = goal_functional_solid(
            solid,
            solid_timeline,
            param,
        )

    print(f"Value of goal functional: {sum(goal_functional)}")
    residuals_txt.write(
        f"Value of goal functional: {sum(goal_functional)} \r\n"
    )
    residuals_txt.close()

    fluid_residual = 0
    for i in range(len(primal_fluid)):
        fluid_residual += abs(primal_fluid[i] + adjoint_fluid[i])
    print(f"Value of fluid residual: {fluid_residual}")

    solid_residual = 0
    for i in range(len(primal_solid)):
        solid_residual += abs(primal_solid[i] + adjoint_solid[i])
    print(f"Value of solid residual: {solid_residual}")

    partial_residual_fluid = open("partial_residual_fluid.txt", "a+")
    partial_residual_solid = open("partial_residual_solid.txt", "a+")
    for i in range(len(primal_fluid)):
        partial_residual_fluid.write(
            f"{abs(primal_fluid[i] + adjoint_fluid[i])}\r\n"
        )
    for i in range(len(primal_solid)):
        partial_residual_solid.write(
            f"{abs(primal_solid[i] + adjoint_solid[i])}\r\n"
        )
    partial_residual_fluid.close()
    partial_residual_solid.close()

    partial_residual_fluid = open("partial_residual_fluid.txt", "r")
    partial_residual_solid = open("partial_residual_solid.txt", "r")
    residual_fluid_test = []
    for x in partial_residual_fluid.read().splitlines():
        residual_fluid_test.append(float(x))
    residual_solid_test = []
    for x in partial_residual_solid.read().splitlines():
        residual_solid_test.append(float(x))
    partial_residual_fluid.close()
    partial_residual_solid.close()

    # [print(x) for x in goal_functional]

    return [
        primal_fluid,
        primal_solid,
        adjoint_fluid,
        adjoint_solid,
        sum(goal_functional),
    ]
