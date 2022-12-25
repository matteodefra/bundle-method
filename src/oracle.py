import numpy as np
from collections import deque
import cvxpy as cp
from mpi4py import MPI

from typing import Tuple, Dict


class Oracle:

 '''
  Class constructor
  - N_Solvers: number of solvers
  - K: the list of components k \in \Mck managed by this oracle
  - queues: N_Solvers deques, one for each solver
  - functions: list of functions to evaluate
 '''
 def __init__(self, algorithms: int, components: int, index_functions: Dict):
  self.N_Solvers = algorithms
  self.K = components
  self.queues = [ [ deque([]) ] * self.K for _ in range(self.N_Solvers) ]
  self.index_functions = index_functions
  self.accuracy = float("inf")
  self.tol = float("inf") 

  # self.comm = MPI.COMM_WORLD
  # self.rank = comm.Get_rank()

  return

 '''
  Print all the queues content
 '''
 def get_queues(self):
  for i in range(self.N_Solvers):
   for j in range(self.K):
    print( self.queues[i][j] )
 
 '''
  push: allows the different Solvers to push values into
  the queues of the Oracle 
 '''
 def push(self, point, caller, component):
  self.queues[caller][component].appendleft(point)
  return

 '''
  pop: the oracle itself removing items from queue to 
  process them
 '''
 def pop(self, queue : deque):
  return queue.pop()

 '''
  Return the evaluation of f at a given point x
 '''
 def f(self, var_x, function: Tuple):
  # Get the constant/vector/matrix
  term = function[1]
  # Evaluate function
  return function[2]( term, var_x )

 def g(self, var_x, function: Tuple):
  term = function[1]
  return function[3]( term, var_x )

 '''
  compute(): called whenever a point x_i is extracted from 
  any queue and
   - f(x_i) or [ underline{f}, bar{f} ] are computed
   - g in { partial f(x_i) } is computed
 '''
 def compute(self, var_x: np.array, component: int, accuracy: float, tolerance: float, caller: int) -> Tuple[ float, np.array ]:
  # Get function definition
  f_x = self.index_functions[ component ]
  # Compute the exact value function at var_x
  f_val = self.f( var_x=var_x, function=f_x )
  g_val = self.g( var_x=var_x, function=f_x )
  # Compute subgradient values of the Lagrangian relaxation

  return (f_val, g_val)

 '''
  run(): launch the oracle waiting for possible values in the
		queues. Round robin policy running through the queues for 
  possible values
	'''
 def run(self):
  while True:



   for k in range(self.K):
    for sol in range(self.N_Solvers):

     # Something is in the queue
     if self.queues[sol][k]:

      self.compute(self.pop(self.queues[sol][k]), k, self.accuracy, self.tol)



  return
