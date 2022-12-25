import cvxpy as cp
import numpy as np
import time
import sys
from mpi4py import MPI

from oracle import Oracle
from PBM import ProximalBundleMethod
from utils import *

# comm = MPI.COMM_SELF.Spawn(sys.executable, ["oracle.py"], maxprocs=2)
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# print(rank)
# print(size)

recover = False

if recover:
	DICT_DATA, N_SOLVERS, COMPONENTS, TIME_PERIOD, index_functions, tuples, demands \
																																						= recover_data("data/autogen/N=10_K=8_T=24")
else:
	N_SOLVERS, COMPONENTS, TIME_PERIOD, index_functions, tuples, demands \
																																						= initialize_vars()

NUM_ORACLES = 10

# if rank == 0:
# 	for i in range(size):
# 		comm.send( DICT_DATA, dest=i, tag=11 )

oracle = Oracle( N_SOLVERS, COMPONENTS, index_functions=index_functions )

# Starting feasible x
x_0 = np.array( [[8] * TIME_PERIOD for _ in range(COMPONENTS)], dtype=float )

# Instantiate Proximal Bundle Method
pbm = ProximalBundleMethod(
	n=COMPONENTS, m=TIME_PERIOD, 
	t=1, type=min, m1=0.5, m2=0.5,
	initial_point=x_0, lb_ub=tuples,
	demands=demands,
	index_functions=index_functions
)

# Just for the sake of debugging
solver_index = 0

# Checking if the oracle works
for k in range(COMPONENTS):
	f, g = oracle.compute( x_0[k, :].T, k, 0, 0, solver_index )
	print( f"f^{k}: {f}" )
	print( f"g in f^{k}: {g}" )

	# Add linearization to the PBM bundle
	cut = pbm.add_cut( f, x_0[k, :].T, g, k )
	# Add also "linearization" for the MP solution
	pbm.add_alt_cut( f, x_0[k, :].T, g, k, cut )

# Compute value function and the model value on the initial iterate
pbm.set_best_values( pbm.true_function( x_0 ), pbm.model( x_0 ) )

# Set a number of steps to perform
step = 0

# for step in range(steps):
while pbm.check_optimality() != "Success":
	# Compute the new candidate point
	new_candidate_x = pbm.step()
	# Check SS/NS
	pbm.check_SS( new_candidate_x, step )
	# For every component function 
	for k in range(COMPONENTS):
		# Invoke the oracle, and then add the cut to the PBM Bundle
		f, g = oracle.compute( new_candidate_x[k, :].T, k, 0, 0, solver_index )
		cut = pbm.add_cut( f, new_candidate_x[k, :].T, g, k )
		pbm.add_alt_cut( f, new_candidate_x[k, :].T, g, k, cut )
	# Logging
	pbm.print_stats(step)
	step += 1
	if step == 100:
		break
	# Check optimality (to move above)

# print(f"Optimal solution achieved: {pbm.best_obj} at iteration {step+1}")

# Termination: print final results
print("\n\n\n")
print(f"Optimal solution: {pbm.best_obj}")
print(f"Time required: {pbm.compute_time}")
print("\n\n\n")

#### Problem solution with cvxpy for the sake of comparison
# Instantiate
x = cp.Variable( shape=(COMPONENTS, TIME_PERIOD) )

# Build the optimization function with cvxpy atoms
f = 0
for key, fun in index_functions.items():
	if fun[0] == "norm":
		f += 0.5 * ( cp.norm2( x[key, :] ) ** 2 )
	elif fun[0] == "quadform":
		f += 0.5 * cp.quad_form( x[key, :], fun[1] )
	else:
		f += ( x[key, :] @ fun[1] )

# Prepare objective
objective = cp.Minimize( f )

# Create list of constraints with the previous generated lb/ub
contrants = []

# Append to the constraint list
for i in range(COMPONENTS):
	for j in range(TIME_PERIOD):
		t = tuples[i,j]
		contrants.append( x[i,j] >= t[0] )
		contrants.append( x[i,j] <= t[1] )

for j in range(TIME_PERIOD):
	contrants.append( sum( x[:,j] ) == demands[j] )

# Instantiate problem
prob = cp.Problem( objective, contrants )

start = time.time()
# Launch cvxpy solver
prob.solve(solver=cp.CPLEX, verbose=False)
end = time.time()

print("\n\n\n")
print(f"Problem status: {prob.status}")
print(f"Optimal solution: {prob.value}")
print(f"Time required: {end-start}")
print("\n\n\n")
