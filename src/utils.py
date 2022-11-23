import numpy as np
import cvxpy as cp


def quadform( P: np.array, x: np.array ) -> float:
 return 0.5 * np.dot( x, P @ x.T )

def d_quadform( P: np.array, x: np.array ) -> np.array:
 return (P @ x.T).T

def dot( c: np.array, x: np.array ) -> float:
 return (x @ c).item()

def d_dot( c: np.array, x: np.array ) -> np.array:
 return c.T.reshape(-1)

def norm( scalar, x: np.array ) -> float:
 return 0.5 * ( np.linalg.norm(x) ** 2 )

def d_norm( scalar, x: np.array ) -> np.array:
 return x