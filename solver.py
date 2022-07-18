import itertools as it

from numpy.random import default_rng
from ortools.sat.python import cp_model

from mindist import minimum_distance

def random_graph(num_qubits, num_stabs, density, rng=default_rng()):
    """
    Generate a random bipartite graph as a list of stabilizer adjacency.
    Each edge is added with probability `density`.
    """
    stabs = [[] for _ in range(num_stabs)]
    for qubit, stab in it.product(range(num_qubits), range(num_stabs)):
        if rng.random() <= density:
            stabs[stab].append(qubit)
    return stabs


class CodeGenerator:
    """A wrapper around a CpModel from ortools."""
    def __init__(self, num_qubits, stabs):
        self.model = cp_model.CpModel()
        self.num_qubits = num_qubits
        self.num_stabs = len(stabs)
        self.stabs = stabs
        self.init_qubit_adjacency()
        self.init_activators()
        self.init_paulis()
        self.init_x_and_z_edges()
        self.extra_vars = list()

    def init_qubit_adjacency(self):
        """Inverse self.stabs"""
        self.qubits = [[] for _ in range(self.num_qubits)]
        for stab, qubits in enumerate(self.stabs):
            for qubit in qubits:
                self.qubits[qubit].append(stab)

    def init_activators(self):
        """Create an activator variable for each edge."""
        self.activators = dict()
        for qubit, stab in self.edges():
            self.activators[(qubit, stab)] = self.model.NewBoolVar(
                f"acti[q = {qubit}, s = {stab}]"
            )

    def init_paulis(self):
        """Create a Pauli variable for each stabilizer."""
        self.paulis = dict()
        for stab in range(self.num_stabs):
            self.paulis[stab] = self.model.NewBoolVar(f"pauli[s = {stab}]")

    def init_x_and_z_edges(self):
        """Create the X and Z edge variables."""
        self.x_edges = dict()
        self.z_edges = dict()
        for (qubit, stab) in self.edges():
            edge = (qubit, stab)
            self.x_edges[edge] = self.model.NewBoolVar(
                f"X_edge[q = {qubit}, s = {stab}]"
            )
            self.z_edges[edge] = self.model.NewBoolVar(
                f"Z_edge[q = {qubit}, s = {stab}]"
            )
            self.add_x_and_z_edge_constraints(
                self.activators[edge],
                self.paulis[stab],
                self.x_edges[edge],
                self.z_edges[edge],
            )

    def add_x_and_z_edge_constraints(self, activator, pauli, x, z):
        """
        Decompose the constraints

            x == activator AND pauli,
            z == activator AND NOT(pauli),

        into the constraints

            Not(x) OR activator,
            Not(z) OR activator,
            Not(activator) OR x OR z,
            Not(x) OR pauli,
            Not(z) OR NOT(pauli),
        """
        self.model.AddBoolOr([x.Not(), activator])
        self.model.AddBoolOr([z.Not(), activator])
        self.model.AddBoolOr([x, z, activator.Not()])
        self.model.AddBoolOr([x.Not(), pauli])
        self.model.AddBoolOr([z.Not(), pauli.Not()])

    def activator_var(self, qubit, stab):
        """Return the corresponding activator variable."""
        return self.activators.get((qubit, stab))

    def pauli_var(self, stab):
        """Return the corresponding Pauli variable."""
        return self.paulis.get(stab)

    def x_edge_var(self, qubit, stab):
        """Return the corresponding X edge variable."""
        return self.x_edges.get((qubit, stab))

    def z_edge_var(self, qubit, stab):
        """Return the corresponding Z edge variable."""
        return self.z_edges.get((qubit, stab))

    def new_extra_var(self):
        """Add an extra variable to the model and return it."""
        var = self.model.NewBoolVar(f"extra[{len(self.extra_vars)}]")
        self.extra_vars.append(var)
        return var

    def add_and_gate(self, left, right, target):
        """
            Decompose the constraint

                left AND right == target

            into the contraints

                Not(target) OR left,
                Not(target) OR right,
                Not(left) OR Not(right) OR target,
        """
        self.model.AddBoolOr([left, target.Not()])
        self.model.AddBoolOr([right, target.Not()])
        self.model.AddBoolOr([left.Not(), right.Not(), target])

    def edges(self):
        """Return an iterator throught all edges of the graph."""
        for (stab, qubits) in enumerate(self.stabs):
            for qubit in qubits:
                yield (qubit, stab)

    def with_commutation_constraints(self):
        """Add the commutation constraints to the model."""
        for (stab0, stab1) in it.combinations(enumerate(self.stabs), r=2):
            self.add_commutation_constraint(stab0, stab1)
        return self

    def add_commutation_constraint(self, stab0, stab1):
        """Add the commutation constraint for the given stabilizers."""
        overlaps = set(stab0[1]) & set(stab1[1])
        if len(overlaps) > 0:
            overlaps_vars = list()
            for qubit in overlaps:
                both_active = self.both_edges_are_active(qubit, stab0[0], stab1[0])
                overlaps_vars.append(both_active)
            parity_var = self.new_extra_var()
            overlaps_vars.append(parity_var)
            self.model.AddBoolXOr(overlaps_vars)
            paulis_commute = self.paulis_commute(stab0[0], stab1[0])
            self.model.AddBoolOr([parity_var, paulis_commute])

    def both_edges_are_active(self, qubit, stab0, stab1):
        """
        Add a variable representing that both edges (qubit, stab0)
        and (qubit, stab1) are actives.
        """
        acti0 = self.activator_var(qubit, stab0)
        acti1 = self.activator_var(qubit, stab1)
        both_active = self.new_extra_var()
        self.add_and_gate(acti0, acti1, both_active)
        return both_active

    def paulis_commute(self, stab0, stab1):
        """
        Add a variable representing that the Pauli values
        of stab0 and stab1 commute.
        """
        pauli0 = self.pauli_var(stab0)
        pauli1 = self.pauli_var(stab1)
        commute = self.new_extra_var()
        self.model.AddBoolXOr([pauli0, pauli1, commute])
        return commute

    def with_min_qubit_deg_constraints(self, min_deg):
        """
        Add the constraint that each qubit is connected to at least
        `min_deg` X stabilizers and `min_deg` Z stabilizers.
        """
        for qubit, stabs in enumerate(self.qubits):
            self.add_min_qubit_x_deg_constraint(qubit, stabs, min_deg)
            self.add_min_qubit_z_deg_constraint(qubit, stabs, min_deg)
        return self

    def add_min_qubit_x_deg_constraint(self, qubit, stabs, min_deg):
        """
        Add the constraint that the given qubit is connected to at least
        `min_deg` X stabilizers.
        """
        x_vars = [self.x_edge_var(qubit, stab) for stab in stabs]
        self.model.Add(sum(x_vars) >= min_deg)

    def add_min_qubit_z_deg_constraint(self, qubit, stabs, min_deg):
        """
        Add the constraint that the given qubit is connected to at least
        `min_deg` Z stabilizers.
        """
        z_vars = [self.z_edge_var(qubit, stab) for stab in stabs]
        self.model.Add(sum(z_vars) >= min_deg)

    def with_min_stab_deg_constraints(self, min_deg):
        """
        Add the constraint that each stabilizer is connected to at least
        `min_deg` qubits.
        """
        for stab, qubits in enumerate(self.stabs):
            self.add_min_stab_deg_constraint(stab, qubits, min_deg)
        return self

    def add_min_stab_deg_constraint(self, stab, qubits, min_deg):
        """
        Add the constraint that the given stabilizer is connected to at least
        `min_deg` qubits.
        """
        vars = [self.activator_var(qubit, stab) for qubit in qubits]
        self.model.Add(sum(vars) >= min_deg)

    def with_max_stab_deg_constraints(self, max_deg):
        """
        Add the constraint that each stabilizer is connected to at most
        `max_deg` qubits.
        """
        for stab, qubits in enumerate(self.stabs):
            self.add_max_stab_deg_constraint(stab, qubits, max_deg)
        return self

    def add_max_stab_deg_constraint(self, stab, qubits, max_deg):
        """
        Add the constraint that the given stabilizer is connected to at most
        `max_deg` qubits.
        """
        vars = [self.activator_var(qubit, stab) for qubit in qubits]
        self.model.Add(sum(vars) <= max_deg)

    def with_balanced_stab_constraint(self):
        """
        Add the constraint that the numbers of X and Z stabilizers
        are the same (or differ by at most 1 if the number of stabilizers is odd).
        """
        vars = [self.pauli_var(stab) for stab in range(self.num_stabs)]
        self.model.Add(sum(vars) == self.num_stabs // 2)
        return self

    def solve(self, max_time_in_seconds, random_seed=None, num_workers=None):
        """Run the solver. See ortools documentation for details."""
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_in_seconds
        if random_seed is not None:
            solver.parameters.random_seed = random_seed
        if num_workers is not None:
            solver.parameters.num_search_workers = num_workers
        status = solver.Solve(self.model)
        return status, solver


def build_stabilizers(generator, solver):
    """
    Return the list of X and Z stabilizers from the solver outcome.
    It assumes that the solver was successful.
    """
    x_stabs = list()
    z_stabs = list()
    for stab, qubits in enumerate(generator.stabs):
        to_keep = list()
        for qubit in qubits:
            if solver.Value(generator.activator_var(qubit, stab)) == 1:
                to_keep.append(qubit)
        if len(to_keep) > 0:
            if solver.Value(generator.pauli_var(stab)) == 1:
                x_stabs.append(sorted(to_keep))
            else:
                z_stabs.append(sorted(to_keep))
    return x_stabs, z_stabs


if __name__ == "__main__":
    generator = (
        CodeGenerator(50, random_graph(50, 45, 0.5))
        .with_commutation_constraints()
        .with_min_qubit_deg_constraints(3)
        .with_min_stab_deg_constraints(6)
        .with_max_stab_deg_constraints(12)
        .with_balanced_stab_constraint()
    )
    status, solver = generator.solve(60)
    print(f"Status is {solver.StatusName(status)}")
    print(f"It took {solver.WallTime()} seconds")
    if solver.StatusName(status) == "OPTIMAL":
        x_stabs, z_stabs = build_stabilizers(generator, solver)
        print(f"Minimum distance is {minimum_distance(50, x_stabs, z_stabs)}")
        print()
        print(f"X stabilizers:")
        for s in x_stabs:
            print(s)
        print()
        print(f"Z stabilizers:")
        for s in z_stabs:
            print(s)
