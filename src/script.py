import cvxpy as cp
import numpy as np
import random

from oracle import Oracle
from PBM import ProximalBundleMethod, Cut	

from utils import *


N_solvers = 10

N_components = 10
hours = 48
# def get_objective():

x = cp.Variable(shape=(N_components, hours))

functions = []

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
		index_functions[i] = ( f, 2 )
	elif f == cp.quad_form:
		# Generate PSD matrix
		A = np.random.rand( hours, hours )
		B = np.dot( A, A.transpose() )
		# append to functions
		index_functions[i] = ( f, B )
	else:
		# Generate random vector
		c = np.random.rand( hours, 1 )
		# append to functions
		index_functions[i] = ( f, c )


oracle = Oracle( N_solvers, N_components, index_functions=index_functions )

x_0 = np.array( [[1] * hours for _ in range(N_components)], dtype=float )

for k in range( N_components ):
	
	if index_functions[k] == cp.quad_form:
		f, g = functions[k].numeric( x_0.T[ :, k, None ] ), functions[k]._grad( x_0.T[ :, k, None ] )
	else:
		f, g = functions[k].numeric( x_0[ k, None, : ] @ l[k] ), l[k]
		

	# if index_functions[k] == cp.quad_form:
	# 	f, g = oracle.compute( x_0.T[:, k, None], k, 0, 0, 0 )
	# else:
	# 	f, g = oracle.compute( x_0[k, None, :], k, 0, 0, 0 )

	# component = random.randint(0, N_components-1)
	print( f"f^{k}: {f}" )
	print( f"True f^{k}: {quadform( x_0.T[ :, k, None ], l2[k] )}" )
	print( f"g in f^{k}: {g}" )
