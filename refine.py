def refine_time(
    primal_fluid,
    primal_solid,
    adjoint_fluid,
    adjoint_solid,
    fluid_timeline,
    solid_timeline,
):

    # Find intervals to refine
    fluid_residuals = [
        abs(x[0] + x[1]) for x in zip(primal_fluid, adjoint_fluid)
    ]
    solid_residuals = [
        abs(x[0] + x[1]) for x in zip(primal_solid, adjoint_solid)
    ]
    average = (
        2.0 * sum([abs(x) for x in fluid_residuals])
        + 2.0 * sum([abs(x) for x in solid_residuals])
    ) / (len(fluid_residuals) + len(solid_residuals))
    fluid_refinements = []
    solid_refinements = []
    for i in range(len(fluid_residuals)):

        if fluid_residuals[i] > average:

            fluid_refinements.append(True)
        else:

            fluid_refinements.append(False)

    for i in range(len(solid_residuals)):

        if solid_residuals[i] > average:

            solid_refinements.append(True)
        else:

            solid_refinements.append(False)

    # Adjust refinement array to preserve patch structure
    for i in range(len(fluid_residuals)):

        if fluid_refinements[i]:

            if i % 2 == 0:

                fluid_refinements[i + 1] = True
            else:

                fluid_refinements[i - 1] = True

    for i in range(len(solid_residuals)):

        if solid_refinements[i]:

            if i % 2 == 0:

                solid_refinements[i + 1] = True
            else:

                solid_refinements[i - 1] = True

    # Save refinement arrays
    fluid_size = fluid_timeline.size_global - fluid_timeline.size
    solid_size = solid_timeline.size_global - solid_timeline.size
    fluid_refinements_txt = open(
        f"fluid_{fluid_size}-{solid_size}_refinements.txt", "a"
    )
    solid_refinements_txt = open(
        f"solid_{fluid_size}-{solid_size}_refinements.txt", "a"
    )
    [fluid_refinements_txt.write(str(int(x))) for x in fluid_refinements]
    [solid_refinements_txt.write(str(int(x))) for x in solid_refinements]
    fluid_refinements_txt.close()
    solid_refinements_txt.close()
