# Model to Partition a DFG across Heterogeneous Execution Engines

The problem of placing and scheduling the nodes of a data-flow graph (DFG) such as a Deep Learning model is an NP-Hard problem.
We pose the problem as an MILP formuluation and solve it using the gurobi solver.

# Decision variables
Placement: *D<sub>i</sub><sup>d</sup>* denotes a binary decision variable which states whether task *i* is to be executed on device *d*.

Scheduling: *P<sub>ij</sub>* denotes a binary decision variables which states whether task *i* needs to be executed before task *j*. 
$\sum_{i,j}{i\neqj}P<sub>ij</sub>$
