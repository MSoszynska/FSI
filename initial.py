from fenics import Function, FunctionSpace, File, HDF5File, MPI

# Define initialization of a function
class Initial:
    def __init__(self, space_name, variable_name, function_space):

        # Define initial values for time loop
        self.new = Function(function_space)
        self.old = Function(function_space)
        self.old_old = Function(function_space)
        self.new_average = Function(function_space)
        self.old_average = Function(function_space)
        self.old_old_average = Function(function_space)

        # Create arrays of empty functions and iterations
        self.array = []
        self.iterations = []

        # Create pvd files
        self.pvd = File(f"solutions/{space_name}/pvd/{variable_name}.pvd")

        # Remember space
        self.function_space = function_space

        # Remember space and variable names
        self.space_name = space_name
        self.variable_name = variable_name

        # Create HDF5 counter
        self.HDF5_counter = 0

    def save(self, function, only_pvd=False):

        # Save solution in pvd format
        function.rename(self.variable_name, self.space_name)
        self.pvd << function

        if not only_pvd:
            # Save solution in HDF5 format
            file = HDF5File(
                MPI.comm_world,
                f"solutions/{self.space_name}"
                f"/HDF5/{self.variable_name}_"
                f"{self.HDF5_counter}.h5",
                "w",
            )
            file.write(
                function,
                f"solutions/{self.space_name}"
                f"/HDF5/{self.variable_name}_{self.HDF5_counter}",
            )
            file.close()
            self.HDF5_counter += 1

        return

    def load(self, number):

        # Load solution in HDF5 format
        function = Function(self.function_space)
        file = HDF5File(
            MPI.comm_world,
            f"solutions/{self.space_name}"
            f"/HDF5/{self.variable_name}_{number}.h5",
            "r",
        )
        file.read(
            function,
            f"solutions/{self.space_name}"
            f"/HDF5/{self.variable_name}_{number}",
        )
        file.close()

        return function
