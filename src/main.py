import cvxpy as cp
import numpy as np
import random

from oracle import Oracle
from PBM import ProximalBundleMethod, Cut	
from utils import *

N_solvers = 10

N_components = 5
hours = 12
# def get_objective():

np.random.seed( 25 )

x = cp.Variable(shape=(N_components, hours))

index_functions = {}

# Sum of convex functions: let choose randomly
list_fun = [ "quadform", "dot", "norm" ]
set_terms = []


for i in range(N_components):
	# Get random function
	fun_index = random.randint( 0, len(list_fun)-1 )

	f = list_fun[fun_index]

	print( f"{i}: {list_fun[fun_index]}" )
	if f == "norm":
		# append 2 for the norm
		index_functions[i] = ( f, 2, norm, d_norm )
	elif f == "quadform":
		# Generate PSD matrix
		A = np.random.rand( hours, hours )
		B = np.dot( A, A.transpose() )
		# append to functions
		index_functions[i] = ( f, B, quadform, d_quadform )
	else:
		# Generate random vector
		c = np.random.rand( hours, 1 )
		# append to functions
		index_functions[i] = ( f, c, dot, d_dot )

tuples = np.zeros( (N_components, hours), dtype='i,i' )

print(tuples)

for i in range(N_components):
	for j in range(hours):
		lb = random.uniform( 1, 5 )
		ub = random.uniform( 8, 12 )
		tuples[i,j] = (lb,ub)

# Instantiate oracle
oracle = Oracle( N_solvers, N_components, index_functions=index_functions )

# Starting feasible x
x_0 = np.array( [[8] * hours for _ in range(N_components)], dtype=float )

pbm = ProximalBundleMethod(
	n=N_components, m=hours, 
	t=1, type=min, m1=0.5, m2=0.5,
	initial_point=x_0, lb_ub=tuples,
	index_functions=index_functions
)

solver_index = 0

for k in range(N_components):
	f, g = oracle.compute( x_0[k, :].T, k, 0, 0, solver_index )
	print( f"f^{k}: {f}" )
	print( f"g in f^{k}: {g}" )

	cut = pbm.add_cut( f, x_0[k, :].T, g, k )
	pbm.add_alt_cut( f, x_0[k, :].T, g, k, cut )

pbm.set_best_values( pbm.true_function( x_0 ), pbm.model( x_0 ) )

x_1 = np.array( [[5] * hours for _ in range(N_components)], dtype=float )

print(pbm.check_SS(x_1, 0))

for k in range(N_components):
	f, g = oracle.compute( x_1[k, :].T, k, 0, 0, solver_index )
	print( f"f^{k}: {f}" )
	print( f"g in f^{k}: {g}" )

	cut = pbm.add_cut( f, x_1[k, :].T, g, k )
	pbm.add_alt_cut( f, x_1[k, :].T, g, k, cut )

pbm.set_best_values( pbm.true_function( x_1 ), pbm.model( x_1 ) )

print( f"Best fun value: { pbm.best_obj }" )
print( f"Best model value: { pbm.best_model_obj }" )

# for k in range(N_components):
# 	f, g = oracle.compute( x_0[k, :].T, k, 0, 0, solver_index )
# 	pbm.add_alt_cut( x_0[k, :].T, g, k )

# pbm.print_functions()
# pbm.print_cuts()
# pbm.print_alt_cuts()

steps = 500

for step in range(steps):
	new_candidate_x = pbm.step()
	pbm.check_SS( new_candidate_x, step )
	for k in range(N_components):
		f, g = oracle.compute( new_candidate_x[k, :].T, k, 0, 0, solver_index )
		cut = pbm.add_cut( f, new_candidate_x[k, :].T, g, k )
		pbm.add_alt_cut( f, new_candidate_x[k, :].T, g, k, cut )
	pbm.print_stats(step)
	if pbm.check_optimality() == "Success":
		print(f"Optimal solution achieved: {pbm.best_model_obj} at iteration {step+1}")
		break
print("\n\n\n")
print(f"Optimal solution: {pbm.best_obj}")
print(f"Time required: {pbm.compute_time}")
print("\n\n\n")

#### Problem solution with cvxpy
x2 = cp.Variable( shape=(N_components, hours) )

f = 0
for key, fun in index_functions.items():
	if fun[0] == "norm":
		f += 0.5 * cp.norm( x2[key, :] ) ** 2
	elif fun[0] == "quadform":
		f += 0.5 * cp.quad_form( x2[key, :], fun[1] )
	else:
		f += ( x2[key, :] @ fun[1] )

objective = cp.Minimize( f )

contrants = []

for i in range(N_components):
	for j in range(hours):
		t = tuples[i,j]
		contrants.append( x2[i,j] >= t[0] )
		contrants.append( x2[i,j] <= t[1] )

prob = cp.Problem( objective, contrants )

prob.solve(verbose=False)

print(prob.status)
print(prob.value)

# print(objective)

# prob = cp.Problem( objective, constraints )

# prob.solve(verbose=True)

# print(prob.status)
# print(prob.value)
# print(x.value)