""" the Solow model with a time-varying savings rate (Problem 2)

Starting point for the exam. The methods that are already implemented are taken
from Lecture 9. The methods raising NotImplementedError are the ones you should
write yourself.

"""

from types import SimpleNamespace
from scipy import optimize
import numpy as np

class SolowModelClass:
    """ the Solow model with a Cobb-Douglas production function """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. technology
        par.alpha = 1/3 # capital share in production
        par.delta = 0.30 # depreciation rate

        # b. the long-run savings rate
        par.s_bar = 0.25

        # c. preferences
        par.beta = 1/1.2 # discount factor

        # d. simulation settings
        par.k0 = 0.10 # initial capital per worker
        par.T = 100 # number of periods

        # e. overwrite with keyword arguments, e.g. SolowModelClass(alpha=0.25)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Solow model with:\n'
        text += f'  alpha = {par.alpha:.4f} (capital share)\n'
        text += f'  delta = {par.delta:.4f} (depreciation rate)\n'
        text += f'  s_bar = {par.s_bar:.4f} (long-run savings rate)\n'
        text += f'  beta  = {par.beta:.4f} (discount factor)\n'
        text += f'  k0    = {par.k0:.4f} (initial capital per worker)\n'
        text += f'  T     = {par.T} (number of periods)'

        return text

    def f(self,k):
        """ output per worker, y = f(k) = k**alpha """

        return k**self.par.alpha

    def k_next(self,k,s):
        """ capital per worker next period, k_next = s*f(k) + (1-delta)*k """

        return s*self.f(k) + (1-self.par.delta)*k

    def steady_state(self,s=None):
        """ analytical steady state for a constant savings rate, returns (k,y,c) """

        par = self.par
        if s is None: s = par.s_bar

        k = (s/par.delta)**(1/(1-par.alpha))
        y = self.f(k)
        c = (1-s)*y

        return k,y,c

    def solve_steady_state(self,s=None):
        """ numerical steady state, the k where k_next(k,s)-k = 0 """

        if s is None: s = self.par.s_bar

        obj = lambda k: self.k_next(k,s)-k # zero in the steady state
        result = optimize.root_scalar(obj,bracket=[1e-8,1e6],method='brentq')

        return result.root

    def simulate(self,s,k0=None):
        """ simulate the model forward in time

        Args:

            s (float or ndarray): savings rate, a number (constant over time) or
                an array with par.T elements (time-varying)
            k0 (float,optional): initial capital per worker, par.k0 is used if None

        Returns:

            (SimpleNamespace): the simulated paths, also stored in self.sim

        """

        par = self.par
        sim = self.sim

        if k0 is None: k0 = par.k0

        # a. savings rate in each period (np.ndim(s) == 0 means s is a single number)
        s_vec = np.full(par.T,s) if np.ndim(s) == 0 else np.asarray(s,dtype=float)
        assert s_vec.size == par.T, f'the savings rate must have {par.T} elements, but has {s_vec.size}'

        # b. allocate memory
        k = np.empty(par.T) # capital per worker
        y = np.empty(par.T) # output per worker
        i = np.empty(par.T) # investment per worker
        c = np.empty(par.T) # consumption per worker

        # c. loop forward in time
        k[0] = k0
        for t in range(par.T):

            y[t] = self.f(k[t]) # production
            i[t] = s_vec[t]*y[t] # investment
            c[t] = y[t]-i[t] # consumption

            if t < par.T-1: k[t+1] = i[t] + (1-par.delta)*k[t] # law of motion

        # d. store the results
        sim.s = s_vec
        sim.k = k
        sim.y = y
        sim.i = i
        sim.c = c

        return sim

    #The savings path s_t = s_bar + (s0-s_bar)*phi**t
    def s_path(self,s0,phi):

        #The savings-rate path for t = 0,...,T-1 
        par = self.par
        t = np.arange(par.T)

        return par.s_bar + (s0-par.s_bar)*phi**t

    #The discounted sum of log(c_t)
    def welfare(self,c):
        #Calculate W = sum beta**t*log(c_t)

        par = self.par
        c = np.asarray(c,dtype=float)
        assert c.size == par.T, f'consumption must have {par.T} elements, but has {c.size}'
        if np.any(c <= 0): return -np.inf #log utility is infeasible at non-positive consumption

        t = np.arange(par.T)

        return np.sum(par.beta**t*np.log(c))

    #Welfare of the savings rule (s0,phi)
    def evaluate(self,s0,phi):
        #Simulate the savings rule (s0,phi) and return its welfare

        s = self.s_path(s0,phi)
        sim = self.simulate(s)

        return self.welfare(sim.c)
