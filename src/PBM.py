import cvxpy as cp
import numpy as np

from typing import Dict, Tuple

import time


class Cut:

 def __init__(self, subgrad, lin_error):

  self.subgrad = subgrad
  self.lin_error = lin_error


class ProximalBundleMethod:

 def __init__(self, n: int, m: int, t: float, type, 
  m1: float, m2: float, initial_point: np.array, 
  lb_ub: Tuple, index_functions: Dict):

  self.n = n
  self.m = m

  self.compute_time = 0

  self._x = cp.Variable((n,m))
  self.v = cp.Variable(n)

  self.t = 1

  self.type = type

  self.constraints = self.build_constraints(lb_ub)

  self.index_functions = index_functions

  self.m1 = m1
  self.m2 = m2

  if 0.0 > m1 or m1 > 1.0:
   print("Armijo parameter m1 must be in range (0,1)")
   exit(1)
  if 0.0 > m2 or m2 > 1.0:
   print("Armijo parameter m2 must be in range (0,1)")
   exit(1)


  self._cuts = [[] for _ in range(self.n)]
  self._center = initial_point
  self._translatedconstraints = [[] for _ in range(self.n)]

  self._dual_values = [[] for _ in range(self.n)]

  self.model_values = [ - float("inf") for _ in range(self.n) ]

  self.best_obj = None
  self.best_model_obj = None
  self.z_star = None
  self.v_star = None


 def build_constraints( self, tuples ):
  constraints = []
  for i in range(self.n):
   for j in range(self.m):
    t = tuples[i,j]
    constraints.append( self._x[i, j] >= t[0] )
    constraints.append( self._x[i, j] <= t[1] )
  return constraints


 def set_best_values( self, best_obj, best_model_obj ):
  if self.best_obj is None or ( best_obj <= self.best_obj and self.type == min ) \
   or (best_obj >= self.best_obj and self.type == max):
   self.best_obj = best_obj
  if self.best_model_obj is None or ( best_model_obj <= self.best_model_obj and self.type == min ) \
   or (best_model_obj >= self.best_model_obj and self.type == max):
   self.best_model_obj = best_model_obj


 def print_cuts(self):
  for k in range(self.n):
   for cut in self._cuts[k]:
    print(f"sub_grad norm: |{np.linalg.norm(cut.subgrad)}|, lin_error: {cut.lin_error}")

 def print_alt_cuts(self):
  for k in range(self.n):
   for cut in self._translatedconstraints[k]:
    print(f"cut: {cut}")

 def print_functions(self):
  for f in self.index_functions:
   print(f[0])

 '''
  Evaluate a given cut:
   simply returns the linearization computed at the given point
 '''
 def evaluate_cut(self, cut: Cut, point: np.array) -> float:
  return (cut.subgrad @ point - cut.lin_error)

 '''
  Add a series of cuts to the list of cuts
 '''
 def add_cut(self, f : float, x: np.array, g: np.array, component: int):
  # if self._center is None:
  alpha = g @ x - f
  cut = Cut( g, alpha )
  self._cuts[component].append(cut)
  return cut

 def add_alt_cut(self, f: float, x: np.array, g: np.array, component: int, cut: Cut):
  # value = f - g @ ( x - self._x[component, :] ) #- cut.lin_error
  value = g @ self._x[component, :] - cut.lin_error
  if self.type == max:
   self._translatedconstraints[component].append( self.v[component] <= value )
  else:
   self._translatedconstraints[component].append( self.v[component] >= value )

 '''
  Auxiliary function to compute the best model value in a point: 
   Return the maximum cut for every component k, and sum everything together
 '''
 def model(self, candidate_point: np.array) -> float:
  total = 0
  for k in range(self.n):
   values = []
   for cut in self._cuts[k]:
    values.append( self.evaluate_cut( cut, candidate_point[k,:] ) )
   if values:
   # if len(list(val)) > 0:
    max_val = max(values)
    if max_val >= self.model_values[k]:
     # Update best f^k
     self.model_values[k] = max_val
    total += max_val
   else:
    total += self.model_values[ k ]
   # if not val:
   #  total += 0
   # total += max( map( lambda cut : self.evaluate_cut(cut, candidate_point[k, :]), self._cuts[k] ) )
  return total

 '''
  Compute the true function value f = \sum f^k
 '''
 def true_function(self, candidate_point: np.array) -> float:
  total = 0
  for k in range(self.n):
   f_x = self.index_functions[k]
   total += f_x[2]( f_x[1], candidate_point[k, :] )
  return total


 def check_SS(self, candidate_point: np.array, step: int) -> bool:

  # new_val = self.true_function( candidate_point ) - self.best_obj
  # target = self.m1 * ( self.model( candidate_point ) - self.best_obj )
  if self.v_star is not None:
   new_val = self.true_function( candidate_point )
   target = self.model( candidate_point ) - self.m2 * self.v_star
  else:
   new_val = self.true_function( candidate_point ) - self.best_obj
   target = self.m1 * ( self.model( candidate_point ) - self.best_obj )

  print(f"New value: {new_val}")
  print(f"Target: {target}")

  if new_val <= target:
   self._center = candidate_point
   self.best_obj = self.true_function( candidate_point )
   self.best_model_obj = self.model( candidate_point )
   print(f"Iter {step + 1}: Serious step performed")
   return True
  else:
   print(f"Iter {step + 1}: Null step")
   return False

 def store_dual_values(self):
  for j in range(len(self._translatedconstraints)):
   component_dual_values = []
   l = self._translatedconstraints[j]
   N = len(l)
   for i in range(N):
    component_dual_values.append( l[i].dual_value )
   self._dual_values[j].clear()
   self._dual_values[j] = component_dual_values

 def compute_different_measures(self):
  # Compute optimal z^* and alpha^*
  z_s = []
  alpha_s = []
  for j in range(len(self._dual_values)):
   convex_combinators = self._dual_values[j]
   cuts_component = self._cuts[j]
   tot = 0
   tot_alpha = 0
   for i in range(len(cuts_component)):
    tot += convex_combinators[i] * cuts_component[i].subgrad
    tot_alpha += convex_combinators[i] * cuts_component[i].lin_error
   z_s.append( tot )
   alpha_s.append( tot_alpha )
  self.z_star = sum( z_s )
  self.alpha_star = sum( alpha_s )
  self.d_star = - self.t * sum( z_s )
  self.v_star = sum( self.v.value )
  self.Delta_star = ( ( self.t / 2 ) * ( np.linalg.norm( self.z_star ) ** 2 ) ) + self.alpha_star

  # if self.L > 200:
  #  # Aggregate cuts
  self.aggregate_cuts( z_s, alpha_s )

 def aggregate_cuts( self, z_s, alpha_s, ):
  for i in range(len(z_s)):
   self._cuts[i] = []
   self._cuts[i] = [ Cut( z_s[i], alpha_s[i] ) ]
   self._translatedconstraints[i] = []
   self._translatedconstraints[i].append(self.v[i] >= z_s[i] @ self._x[i,:] - alpha_s[i])

 def check_optimality(self) -> str:
  if self.z_star is not None and self.alpha_star is not None:
    z_star_norm = np.linalg.norm( self.z_star )
    alpha_star_norm = np.linalg.norm( self.alpha_star )
    if z_star_norm <= 1e-20 and alpha_star_norm <= 1e-20:
     return "Success"
  return ""

 def print_stats(self, step):
  tot = 0
  for l in self._cuts:
   tot += len(l)
  print( f"Iter {step + 1}\n\tBest f(x): {self.best_obj}\n\tBest f^(x): {self.best_model_obj} \
   \n\t|z*|: {np.linalg.norm( self.z_star )}\n\talpha*: {self.alpha_star}\n\tv*: {self.v_star} \
   \n\tDelta*: {self.Delta_star}\n\t|cuts|: {tot}" )
  

 '''
  Performs a step of the PBM: it solves the master problem
  with the lower model of f, and return the candidate iterate
 '''
 def step(self) -> np.array:

  if self.type == max:
   obj = cp.Maximize( cp.sum(self.v) - 0.5 * (1/self.t) * cp.sum_squares( self._x - self._center ) )
  else:
   obj = cp.Minimize( cp.sum(self.v) + 0.5 * (1/self.t) * cp.sum_squares( self._x - self._center ) )

  # if self.z_star is None:
  #  self.t = self.t / 2
  # else:
  #  self.t = (self.best_obj - ( self.best_obj - 5)) / ( 0.5 * np.linalg.norm(self.z_star)**2 ) # self.t / (1.5)
  self.t = self.t / 2

  all_constraints = []

  for l in self._translatedconstraints:
   all_constraints.extend( l )

  prob = cp.Problem( obj, self.constraints + all_constraints )
  
  start = time.time()
  prob.solve(verbose=False)
  end = time.time()
  self.compute_time += (end - start) 

  '''
   Refine constraints based on the "active set" of variables:
    Sweep through the inequality constraints -> if the Lagrangian variable 
    tends to zero, then by complementary slackness the value is optimum, hence
    the constraint can be removed.
  ''' 
  # self.L = 0
  # idx = [[] for _ in range(self.n)] 
  # for j in range(len(self._translatedconstraints)):
  #  l = self._translatedconstraints[j]
  #  for i, c in enumerate(l):
  #   # if i == 0:
  #   #  idx[j].append(i)
  #   # elif i != 0 and c.dual_value > 1:
  #   #  idx[j].append(i)
  #   idx[j].append(i)
  #  self.L += len(idx[j])

  # # print(f"Filtering: {self.L}")

  # new_translated_constraints = [[] for _ in range(self.n)]

  # for j in range(len(idx)):
  #  l = idx[j]
  #  for i in l:
  #   new_translated_constraints[j].append( self._translatedconstraints[j][i] )

  # self._translatedconstraints = new_translated_constraints

  # new_cuts = [[] for _ in range(self.n)]

  # for j in range(len(idx)):
  #  l = idx[j]
  #  for i in l:
  #   new_cuts[j].append( self._cuts[j][i] )

  # self._cuts = new_cuts

  self.store_dual_values()

  self.compute_different_measures()

  return self._x.value
 