import numpy as np
import pickle
import random

from utils import *
## HELPER TO GENERATE PICKLE OBJECTS

# Instantiate number of Solvers
N_SOLVERS = 20

# Instantiate number of functions components K and time horizon
COMPONENTS = 16
TIME_PERIOD = 72

# Seed for random number generator
np.random.seed( 12 )

# Dictionary for storing functions definition
# Each entry is defined by:
#  key: index of the function component | val: Tuple of 4 values ( f, value, function, derivative )
#		f = function string representation
# 	value = scalar/vector/matrix (eventually) involved in the function definition
#		function = function computing f(x)
#  derivative = function computing \partial f(x)
index_functions = {}

# List of functions to choose from
list_fun = [ "norm", "dot" ]


# Sweep through the components
for i in range(COMPONENTS):

 # Get random function from list of functions
 fun_index = random.randint( 0, len(list_fun)-1 )
 f = list_fun[fun_index]

 # Discretize based on the chosen function
 if f == "norm":
  # append 2 for the norm
  index_functions[i] = ( f, 2, norm, d_norm )
 elif f == "quadform":
  # Generate PSD matrix
  A = np.random.rand( TIME_PERIOD, TIME_PERIOD )
  B = np.dot( A, A.transpose() )
  # append to functions
  index_functions[i] = ( f, B, quadform, d_quadform )
 else:
  # Generate random vector
  c = np.random.rand( TIME_PERIOD, 1 )
  # append to functions
  index_functions[i] = ( f, c, dot, d_dot )

# Array storing the demand constraint values, i.e. \sum_{i} x_{i,j} == d_j \forall j
demands = np.zeros( TIME_PERIOD )

for j in range(TIME_PERIOD):
 demands[j] = random.uniform( 5, 8 ) * COMPONENTS

# Tuples storing the capacity constraints values, i.e. lb <= x_{i,j} <= ub
tuples = np.zeros( (COMPONENTS, TIME_PERIOD), dtype='i,i' )

# Sweep through all variables and randomly generate the thresholds
for i in range(COMPONENTS):
 for j in range(TIME_PERIOD):
  lb = random.uniform( 1, 5 )
  ub = random.uniform( 8, 12 )
  tuples[i,j] = (lb,ub)

dictionary = {
 "SOLVERS": N_SOLVERS,
 "COMPONENTS": COMPONENTS,
 "TIME_PERIOD": TIME_PERIOD,
 "FUNCTIONS": index_functions,
 "CAPACITY": tuples,
 "DEMAND": demands
}

filename = f"data/autogen/N={N_SOLVERS}_K={COMPONENTS}_T={TIME_PERIOD}"
outfile = open( filename, "wb" )

pickle.dump( dictionary, outfile )
outfile.close()