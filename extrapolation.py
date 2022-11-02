from math import log

fsi1 = True
fsi2 = False
fsi3 = False
uniform_equal = False
uniform_refined = False
adaptive = False

primal_residual_fluid = []
primal_residual_solid = []
adjoint_residual_fluid = []
adjoint_residual_solid = []
functional_extrapolation = []
functional = []

if fsi1:
    functional_extrapolation.append(2.080759992132676)
    functional_extrapolation.append(2.151928291621913)
    functional_extrapolation.append(2.1717347420819975)
    functional_extrapolation.append(2.1773250076724966)

if fsi2:
    functional_extrapolation.append(4.040191214320071)
    functional_extrapolation.append(4.1457793689135904)
    functional_extrapolation.append(4.345414708494079)
    functional_extrapolation.append(4.411133500035451)
    functional_extrapolation.append(4.4349050963650845)

if fsi3:
    functional_extrapolation.append(5.5880389841520985)
    functional_extrapolation.append(5.61830334450651)
    functional_extrapolation.append(5.625529688668127)
    functional_extrapolation.append(5.627186183710912)

if fsi1:

    primal_residual_fluid.append(0.02967583574253895)
    primal_residual_fluid.append(0.009358849260084597)
    primal_residual_fluid.append(0.002458147760512557)
    primal_residual_fluid.append(0.0006303129008203095)

    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)

    adjoint_residual_fluid.append(0.1122725048488098)
    adjoint_residual_fluid.append(0.00886991647921428)
    adjoint_residual_fluid.append(0.0020543352783122756)
    adjoint_residual_fluid.append(0.0005374724519269301)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(2.0808263391549233)
    functional.append(2.151928291621913)
    functional.append(2.1717347420819975)
    functional.append(2.1773250076724966)

if fsi2:
    primal_residual_fluid.append(5.11967730971732)
    primal_residual_fluid.append(0.36184731745762955)
    primal_residual_fluid.append(0.02846523738014132)
    primal_residual_fluid.append(0.007889368467911906)
    primal_residual_fluid.append(0.002054506028640073)

    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.020118339647534633)
    adjoint_residual_fluid.append(0.0049771851898077965)
    adjoint_residual_fluid.append(0.001348129938371152)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(4.040191214320071)
    functional.append(4.1457793689135904)
    functional.append(4.345414708494079)
    functional.append(4.411133500035451)
    functional.append(4.4349050963650845)

if fsi3:
    primal_residual_fluid.append(0.00298200257561516)
    primal_residual_fluid.append(0.0006759345125026772)
    primal_residual_fluid.append(0.00016834481893154314)
    primal_residual_fluid.append(4.375287391393143e-05)

    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)
    primal_residual_solid.append(0.0)

    adjoint_residual_fluid.append(0.01793445485706852)
    adjoint_residual_fluid.append(0.0018965802496868325)
    adjoint_residual_fluid.append(0.0004533722990695928)
    adjoint_residual_fluid.append(0.00011648814426520281)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(5.5880389841520985)
    functional.append(5.61830334450651)
    functional.append(5.625529688668127)
    functional.append(5.627186183710912)

# Perform extrapolation
print("Extrapolation")
J_exact = 0.0
J = functional_extrapolation
for i in range(len(J) - 2):

    print(f"Extrapolation of J{i + 1}, J{i + 2}, J{i + 3}")
    q = -log(abs((J[i + 1] - J[i + 2]) / (J[i] - J[i + 1]))) / log(2.0)
    print(f"Extrapolated order of convergence: {q}")
    C = pow(J[i] - J[i + 1], 2) / (J[i] - 2.0 * J[i + 1] + J[i + 2])
    print(f"Extrapolated constant: {C}")
    J_exact = (J[i] * J[i + 2] - J[i + 1] * J[i + 1]) / (
        J[i] - 2.0 * J[i + 1] + J[i + 2]
    )
    print(f"Extrapolated exact value of goal functional: {J_exact}")

# Compute effectiveness
print("Effectivity")
J = functional
for i in range(len(J)):

    print(f"Effectivity of J{i + 1}")
    residual = (
        primal_residual_fluid[i]
        + primal_residual_solid[i]
        + adjoint_residual_fluid[i]
        + adjoint_residual_solid[i]
    )
    print(f"Overall residual: {residual}")
    print(f"Extrapolated error: {J_exact - J[i]}")
    effectivity = residual / (J_exact - J[i])
    print(f"Effectivity: {effectivity}")
