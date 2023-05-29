from fenics import Constant
from math import sqrt

# Define parameters
class Parameters:
    def __init__(
        self,
        rho_fluid=Constant(1000.0),
        rho_solid=Constant(1000.0),
        nu=Constant(0.001),
        mu=Constant(2000000.0),
        gamma=Constant(1000.0),
        mean_velocity=2.0,
        initial_time=0,
        time_step=0.0001,
        theta=0.5,
        global_mesh_size=2,
        local_mesh_size_fluid=1,
        local_mesh_size_solid=1,
        tau=Constant(0.99),
        absolute_tolerance_relaxation=1.0e-6,
        relative_tolerance_relaxation=1.0e-6,
        max_iterations_relaxation=100,
        epsilon=1.0e-2,
        absolute_tolerance_newton=1.0e-10,
        relative_tolerance_newton=1.0e-8,
        max_iterations_newton=25,
        tolerance_gmres=1.0e-8,
        max_iterations_gmres=100,
        relaxation=False,
        shooting=True,
        goal_functional_fluid=False,
        goal_functional_solid=True,
        compute_primal=True,
        compute_adjoint=False,
        refinement_levels=0,
        partial_load=0,
        partial_compute=None,  # [starting_point, size]
        initial_counter=0,
        adjoint_test_epsilon=0.0,  # * sqrt(10),
    ):

        # Define problem parameters
        self.RHO_FLUID = rho_fluid
        self.RHO_SOLID = rho_solid
        self.NU = nu
        self.MU = mu
        self.GAMMA = gamma
        self.ZETA = (2.0 * mu * 0.4) / (1.0 - 2.0 * 0.4)
        self.MEAN_VELOCITY = mean_velocity

        # Define time step on the coarsest level
        self.INITIAL_TIME = initial_time
        self.TIME_STEP = time_step
        self.THETA = theta

        # Define number of macro time steps on the coarsest level
        self.GLOBAL_MESH_SIZE = global_mesh_size

        # Define number of micro time-steps for fluid
        self.LOCAL_MESH_SIZE_FLUID = local_mesh_size_fluid

        # Define number of micro time-steps for solid
        self.LOCAL_MESH_SIZE_SOLID = local_mesh_size_solid

        # Define relaxation parameters
        self.TAU = tau
        self.ABSOLUTE_TOLERANCE_RELAXATION = absolute_tolerance_relaxation
        self.RELATIVE_TOLERANCE_RELAXATION = relative_tolerance_relaxation
        self.MAX_ITERATIONS_RELAXATION = max_iterations_relaxation

        # Define parameters for Newton's method
        self.EPSILON = epsilon
        self.ABSOLUTE_TOLERANCE_NEWTON = absolute_tolerance_newton
        self.RELATIVE_TOLERANCE_NEWTON = relative_tolerance_newton
        self.MAX_ITERATIONS_NEWTON = max_iterations_newton

        # Define parameters for GMRES method
        self.TOLERANCE_GMRES = tolerance_gmres
        self.MAX_ITERATIONS_GMRES = max_iterations_gmres

        # Choose decoupling method
        self.RELAXATION = relaxation
        self.SHOOTING = shooting

        # Choose goal functional
        self.GOAL_FUNCTIONAL_FLUID = goal_functional_fluid
        self.GOAL_FUNCTIONAL_SOLID = goal_functional_solid

        # Decide if primal and adjoint problems should be solved
        self.COMPUTE_PRIMAL = compute_primal
        self.COMPUTE_ADJOINT = compute_adjoint

        # Set number of refinement levels
        self.REFINEMENT_LEVELS = refinement_levels

        # Set the level of partial loading or computing of solutions
        self.PARTIAL_LOAD = partial_load
        self.PARTIAL_COMPUTE = partial_compute
        self.INITIAL_COUNTER = initial_counter

        # Set epsilon to test the adjoint
        self.ADJOINT_TEST_EPSILON = adjoint_test_epsilon
