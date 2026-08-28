from types import SimpleNamespace
import numpy as np
from scipy import optimize


class ConsumerClass():
    """ a consumer with nested CES preferences over three goods

    Good 1 is food. Goods 2 and 3 are bus trips and train trips, and they sit
    together in a nest.

    The problem is written in *nested* budget shares, in the same two steps as
    the nests themselves:

        s1 = the share of income spent on food
        w  = the share of the remaining (travel) budget spent on the bus

    so that s2 = (1-s1)*w and s3 = (1-s1)*(1-w). Any (s1,w) in the unit square
    is a possible choice, and every possible choice is in the unit square, so
    the constraint set is exactly the box that L-BFGS-B takes as `bounds`.

    """

    def __init__(self,par=None):

        # a. setup
        self.setup()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        """ set the baseline parameters """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preference weights
        par.alpha = 0.60 # weight on food
        par.beta = 0.50 # weight on the bus

        # b. substitution
        par.sigma_A = 0.80 # between food and travel (upper nest)
        par.sigma_B = 0.40 # between bus and train (lower nest)

        # c. prices and income
        par.p1 = 1.0 # price of food
        par.p2 = 1.0 # price of a bus trip
        par.p3 = 1.5 # price of a train trip
        par.I = 10.0 # income

        # d. numerical settings
        par.s_min = 1e-12 # smallest quantity allowed in .ces(), see the note there

    def __str__(self):
        """ print the parameters """

        par = self.par

        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.4f}, beta = {par.beta:.4f}')
        lines.append(f'  sigma_A = {par.sigma_A:.4f}, sigma_B = {par.sigma_B:.4f}')
        lines.append(f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, p3 = {par.p3:.4f}')
        lines.append(f'  I = {par.I:.4f}')

        return '\n'.join(lines)

    ###################
    # 1. the CES nest #
    ###################

    def ces(self,z1,z2,w,sigma):
        """ the CES aggregate of two inputs

        Computes (w*z1**rho + (1-w)*z2**rho)**(1/rho) with rho = 1-1/sigma.

        The inputs are floored at par.s_min, because z**rho is not defined for
        z = 0 when rho < 0, and the corners of the unit square do give zeros.

        Note on the size of par.s_min. It must be *far* below the step that
        L-BFGS-B uses to estimate the gradient, which is `eps` and about 1e-8 by
        default. Otherwise both z and z+eps get floored to the same number near a
        bound, utility comes out exactly equal in the two points, the estimated
        gradient is zero, and the solver stops on the bound and stays there. With
        s_min = 1e-8 that really happens: the answers in section 4 for high tax
        rates come out as zero revenue. 1e-12 is small enough to be invisible and
        large enough to keep z**rho from overflowing.

        Args:

            z1 (float or ndarray): first input
            z2 (float or ndarray): second input
            w (float): weight on the first input
            sigma (float): substitution parameter, must not be 1

        Returns:

            (float or ndarray): the CES aggregate

        """

        par = self.par

        assert not np.isclose(sigma,1.0), 'sigma = 1 gives rho = 0 and a division by zero'


        z1 = np.maximum(z1,par.s_min)
        z2 = np.maximum(z2,par.s_min)


        rho = 1-1/sigma

        return (w*z1**rho + (1-w)*z2**rho)**(1/rho)

    def utility(self,x1,x2,x3):
        """ nested CES utility of a bundle of quantities

        Two steps: first combine goods 2 and 3 into the travel composite, then
        combine good 1 and the composite into utility. Use .ces() for both.

        Args:

            x1 (float or ndarray): quantity of good 1
            x2 (float or ndarray): quantity of good 2
            x3 (float or ndarray): quantity of good 3

        Returns:

            (float or ndarray): utility

        """

        par = self.par

        #Travel nest for bus and trains
        xB= self.ces(x2,x3,par.beta,par.sigma_B)

        #Nest for food and travel
        u=self.ces(x1,xB,par.alpha,par.sigma_A)


        return u

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self,s1,w):
        """ the three budget shares implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three budget shares, which always sum to one

        """

               

        return s1,(1-s1)*w,(1-s1)*(1-w) #always sums to 1 when s1,w are in the interval [0,1]

    def quantities(self,s1,w):
        """ the quantities implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three quantities

        """

        par = self.par

        s1,s2,s3 = self.shares(s1,w)

        return s1*par.I/par.p1, s2*par.I/par.p2, s3*par.I/par.p3 #Ensures total spending = Income

    def value_of_choice(self,s1,w): 
        """ utility of the bundle implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (float or ndarray): utility

        """
        x1,x2,x3 = self.quantities(s1,w)

        u = self.utility(x1,x2,x3)

        return u #returns utility associated with specific budget choice

    def objective(self,s):
        """ minus utility, for a minimizer

        Nothing else is needed: the bounds are the whole constraint.

        Args:

            s (ndarray): array of length 2 with (s1,w)

        Returns:

            (float): minus utility

        """

        return -self.value_of_choice(s[0],s[1]) #SciPy is for minimizing, therefore, we use -utility to find "max", s[0]=s1, s[1]=w

    #################
    # 3. solving it #
    #################

    def solve_grid(self,N=200,do_print=True):
        """ solve by a 2-dimensional grid search over the nested shares

        Every point of the unit square is a possible choice, so there is nothing
        to mask out here -- a plain np.argmax will do.

        Args:

            N (int): number of grid points for each variable
            do_print (bool): print the solution

        Returns:

            (SimpleNamespace): the grids, the utility values and the best point

        """

        par = self.par
        opt = SimpleNamespace()

        # a. the two grids
        s1_vec=np.linspace(0.0,1.0,N)
        w_vec=np.linspace(0.0,1.0,N)

        #Every combination of candidate values
        s1_grid,w_grid = np.meshgrid(s1_vec,w_vec,indexing='ij')


        # b. utility in every grid point
        u_grid=self.value_of_choice(s1_grid,w_grid)

        
        # c. the best point
        index = np.unravel_index(np.argmax(u_grid),u_grid.shape)

        opt.s1 = s1_grid[index]
        opt.w = w_grid[index]
        opt.s1,opt.s2,opt.s3 = self.shares(opt.s1,opt.w)
        opt.u = u_grid[index]

        opt.s1_grid = s1_grid
        opt.w_grid = w_grid
        opt.u_grid = u_grid


        return opt

    def solve(self,s0=None,do_print=True,**kwargs):
        """ solve with L-BFGS-B

        The bounds are ((0,1),(0,1)) -- the whole constraint set.

        Args:

            s0 (ndarray): starting guess for (s1,w)
            do_print (bool): print the solution
            kwargs: passed on to optimize.minimize, e.g. options={'ftol':1e-10}

        Returns:

            (SimpleNamespace): the solution and the convergence path

        """

        par = self.par
        opt = SimpleNamespace()

        # a. starting guess
        if s0 is None: s0 = np.array([0.5,0.5])
        s0 = np.asarray(s0,dtype=float)

        # b. record the path with a callback
        path = [s0.copy()]

        # c. minimize
        res = optimize.minimize(
            self.objective,
            s0,
            method='L-BFGS-B',
            bounds=((0.0,1.0),(0.0,1.0)),#0<=s1<=1 and 0<=w<=1 
            callback= lambda sk: path.append(sk.copy()),
            **kwargs
        )
        # d. results
        opt.s1 = res.x[0]
        opt.w = res.x[1]
        opt.s1,opt.s2,opt.s3 = self.shares(opt.s1,opt.w)
        opt.u = self.value_of_choice(opt.s1,opt.w)
        opt.path = np.asarray(path)
        opt.res = res

        return opt



class ConsumerClass_drink():
    
    def __init__(self,par=None):

        # a. setup
        self.setup()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        #set the baseline parameters for the four-good model

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preference weights
        par.gamma = 0.85 #weight on food relative to drinks
        par.beta = 0.50 #weight on the bus relative to the train
        par.alpha = 0.60 #weight on consumption relative to travel

        # b. substitution elasticities
        par.sigma_C = 0.40 # food and drinks
        par.sigma_B = 0.40 # bus and train 
        par.sigma_A = 0.80 # consumption and travel (lower nest)

        # c. prices and income
        par.p1 = 1.0 # price of food
        par.p2 = 1.0 # price of a drink
        par.p3 = 1.0 # price of a bus trip
        par.p4 = 1.5 # price of a train trip
        par.I = 10.0 # income

        # d. numerical settings
        par.s_min = 1e-12 # smallest quantity allowed in .ces()

    def __str__(self):
        #print the parameters

        par = self.par

        lines = ['ConsumerClass_drink']
        lines.append(
            f'  gamma= {par.gamma:.4f}, beta = {par.beta:.4f}, '
            f'alpha = {par.alpha:.4f}'
        )
        lines.append(
            f'  sigma_C = {par.sigma_C:.4f}, sigma_B = {par.sigma_B:.4f}, '
            f'sigma_A = {par.sigma_A:.4f}'
        )
        lines.append(
            f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, '
            f'p3 = {par.p3:.4f}, p4 = {par.p4:.4f}'
        )
        lines.append(f'  I = {par.I:.4f}')

        return '\n'.join(lines)

    ####################
    # 1. the CES nests  #
    ####################

    def ces(self,z1,z2,w,sigma):
        #the CES aggregate of two inputs

        par = self.par

        assert not np.isclose(sigma,1.0), \
            'sigma = 1 gives rho = 0 and a division by zero'

        z1 = np.maximum(z1,par.s_min)
        z2 = np.maximum(z2,par.s_min)
        rho = 1-1/sigma

        return (w*z1**rho + (1-w)*z2**rho)**(1/rho)

    def utility(self,x1,x2,x3,x4):
        #nested CES utility of food, drinks, bus trips and train trips

        par = self.par

        #LGroceries nest: food and drinks
        xG = self.ces(x1, x2, par.gamma, par.sigma_C)

        #Lower travel nest: bus and train
        xT = self.ces(x3, x4, par.beta, par.sigma_B)
        #Upper nest: groceries and travel
        u = self.ces(xG,xT,par.alpha,par.sigma_A)


        return u

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self,sG,wF,wB):
        #return the four budget shares, which always sum to one

        s_food = sG*wF
        s_drink = sG*(1-wF)
        s_bus = (1-sG)*wB
        s_train = (1-sG)*(1-wB)

        return s_food,s_drink,s_bus,s_train

    def quantities(self,sG,wF,wB):
        #return quantities implied by the three nested shares
        par = self.par
        s1,s2,s3,s4 = self.shares(sG,wF,wB)

        return (
            s1*par.I/par.p1,
            s2*par.I/par.p2,
            s3*par.I/par.p3,
            s4*par.I/par.p4
        )

    def value_of_choice(self,sG,wF,wB):
        #utility of the bundle implied by the nested shares

        x1,x2,x3,x4 = self.quantities(sG,wF,wB)
        return self.utility(x1,x2,x3,x4)

    def objective(self,s):
        #minus utility, for a minimizer

        return -self.value_of_choice(s[0],s[1],s[2])

    #################
    # 3. solving it #
    #################

    def solve_grid(self,N=50,do_print=True):
        #solve by a 3-dimensional grid search over the nested shares (smaller N, because now we are working with N**3)

        
        opt = SimpleNamespace()

        # a. the three grids
        sG_vec = np.linspace(0.0,1.0,N)
        wF_vec = np.linspace(0.0,1.0,N)
        wB_vec = np.linspace(0.0,1.0,N)
        sG_grid,wF_grid,wB_grid = np.meshgrid(
            sG_vec,wF_vec,wB_vec,indexing='ij'
        )

        # b. utility in every grid point
        u_grid = self.value_of_choice(sG_grid,wF_grid,wB_grid)

        # c. the best point
        index = np.unravel_index(np.argmax(u_grid),u_grid.shape)
        opt.sG = sG_grid[index]
        opt.wF = wF_grid[index]
        opt.wB = wB_grid[index]
        opt.s1,opt.s2,opt.s3,opt.s4 = self.shares(opt.sG,opt.wF,opt.wB)
        opt.u = u_grid[index]

        opt.sG_grid = sG_grid
        opt.wF_grid = wF_grid
        opt.wB_grid = wB_grid
        opt.u_grid = u_grid

        return opt

    def solve(self,s0=None,do_print=True,**kwargs):
        #Solve with L-BFGS-B on the unit cube

        opt = SimpleNamespace()

        # a. starting guess
        if s0 is None: s0 = np.array([0.5,0.5,0.5])
        s0 = np.asarray(s0,dtype=float)

        # b. record the convergence path
        path = [s0.copy()]

        # c. minimize
        res = optimize.minimize(
            self.objective,
            s0,
            method='L-BFGS-B',
            bounds=((0.0,1.0),(0.0,1.0),(0.0,1.0)),
            callback=lambda sk: path.append(sk.copy()),
            **kwargs
        )

        # d. results
        opt.sG = res.x[0]
        opt.wF = res.x[1]
        opt.wB = res.x[2]
        opt.s1,opt.s2,opt.s3,opt.s4 = self.shares(opt.sG,opt.wF,opt.wB)
        opt.u = self.value_of_choice(opt.sG,opt.wF,opt.wB)
        opt.path = np.asarray(path)
        opt.res = res

        return opt