"""

Author: Akshay Shanker, Australian National University, akshay.shanker@anu.edu.au

"""


T = int(1e6)




import numpy as np
from scipy.optimize import minimize, brentq, root, fsolve
import scipy.optimize as optimize
from quantecon import MarkovChain
import quantecon.markov as Markov
import quantecon as qe
from numba import jit, vectorize
from pathos.multiprocessing import ProcessingPool
import time
import  dill as pickle
from gini import gini





class ConsumerProblem:
    """
    A class that stores primitives for the income fluctuation problem.  The
    income process is assumed to be a finite state Markov chain.

    Parameters
    ----------
    r : scalar(float), optional(default=0.01)
        A strictly positive scalar giving the interest rate
    Lambda: scalar(float), optional(default = 0.1)
        The shadow social value of accumulation 
    beta : scalar(float), optional(default=0.96)
        The discount factor, must satisfy (1 + r) * beta < 1
    Pi : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
        A 2D NumPy array giving the Markov matrix for {z_t}
    z_vals : array_like(float), optional(default=(0.5, 0.95))
        The state space of {z_t}
    b : scalar(float), optional(default=0)
        The borrowing constraint
    grid_max : scalar(float), optional(default=16)
        Max of the grid used to solve the problem
    grid_size : scalar(int), optional(default=50)
        Number of grid points to solve problem, a grid on [-b, grid_max]
    u : callable, optional(default=np.log)
        The utility function
    du : callable, optional(default=lambda x: 1/x)
        The derivative of u

    Attributes
    ----------
    r, beta, Pi, z_vals, b, u, du : see Parameters
    asset_grid : np.ndarray
        One dimensional grid for assets

    """



    def __init__(self, 
                 r=0.074, 
                 w =.4,
                 Lambda_H = 0,
                 Lambda_E = 0,
                 beta=.945, 
                 Pi=((0.09, 0.91), (0.06, 0.94)),
                 z_vals=(0.1, 1.0), 
                 b= 1e-2, 
                 grid_max= 50, 
                 grid_size= 100,
                 gamma_c = 1.458,
                 gamma_l = 2.833,
                 A_L  = .856):



        
        self.r, self.R = r, 1 + r
        self.w = w
        self.Lambda_H = Lambda_H
        self.Lambda_E= Lambda_E
        self.beta, self.b = beta, b
        self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)
        self.asset_grid = np.linspace(b, grid_max, grid_size)
        self.k = self.asset_grid[1]-self.asset_grid[0] # used for explicit point finding
        self.gamma_c, self.gamma_l = gamma_c, gamma_l
        self.A_L = A_L

        self.u = lambda x, l: np.divide((np.power(x,1-self.gamma_c)-1),(1-self.gamma_c)) +np.multiply(self.A_L,(np.power(l,1-self.gamma_l)-1)/(1-self.gamma_l))
        self.du = lambda x: np.power(x, -self.gamma_c) 
        self.dul = lambda l: A_L*np.power(l, -self.gamma_l) 
        self.grid_max =grid_max
        self.grid_size = grid_size
    

        

class FirmProblem:
    """
    A class that stores primitives for the income fluctuation problem.  The
    income process is assumed to be a finite state Markov chain.

    Parameters
    ----------
    r : scalar(float), optional(default=0.01)
        A strictly positive scalar giving the interest rate
    Lambda: scalar(float), optional(default = 0.1)
        The shadow social value of accumulation 
    beta : scalar(float), optional(default=0.96)
        The discount factor, must satisfy (1 + r) * beta < 1
    Pi : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
        A 2D NumPy array giving the Markov matrix for {z_t}
    z_vals : array_like(float), optional(default=(0.5, 0.95))
        The state space of {z_t}
    b : scalar(float), optional(default=0)
        The borrowing constraint
    grid_max : scalar(float), optional(default=16)
        Max of the grid used to solve the problem
    grid_size : scalar(int), optional(default=50)
        Number of grid points to solve problem, a grid on [-b, grid_max]
    u : callable, optional(default=np.log)
        The utility function
    du : callable, optional(default=lambda x: 1/x)
        The derivative of u

    Attributes
    ----------
    r, beta, Pi, z_vals, b, u, du : see Parameters
    asset_grid : np.ndarray
        One dimensional grid for assets

    """


    def __init__(self, 
                 alpha=1-.64, 
                 delta = .083,
                 AA = 1.51
                        ):


        self.alpha, self.delta, self.AA = alpha, delta, AA


def bellman_operator(V, cp, return_policy=False):
    """
    The approximate Bellman operator, which computes and returns the
    updated value function TV (or the V-greedy policy c if
    return_policy is True).

    Parameters
    ----------
    V : array_like(float)
        A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)
    cp : ConsumerProblem
        An instance of ConsumerProblem that stores primitives
    return_policy : bool, optional(default=False)
        Indicates whether to return the greed policy given V or the
        updated value function TV.  Default is TV.

    Returns
    -------
    array_like(float)
        Returns either the greed policy given V or the updated value
        function TV.

    """
    # === Simplify names, set up arrays === #
    R, w, Lambda_H, Lambda_E, Pi, beta, u, b = cp.R, cp.w, cp.Lambda_H,cp.Lambda_E, cp.Pi, cp.beta, cp.u, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    new_V = np.empty(V.shape)
    new_h = np.empty(V.shape)
    new_l = np.empty(V.shape)
    z_idx = list(range(len(z_vals)))


    # === Linear interpolation of V along the asset grid === #
    #vf = lambda a, i_z: np.interp(a, asset_grid, V[:, i_z])
    vf = lambda a, i_z: np.interp(a, asset_grid, V[:, i_z])

    # === Solve r.h.s. of Bellman equation === #

    def do_bell(i_a):
        a = asset_grid[i_a]
        #print(a)
        for i_z, z in enumerate(z_vals):
            def obj(x):  # objective function to be *minimized*
                y = sum(vf(x[0], j) * Pi[i_z, j] for j in z_idx)
                return - u(R*a +w*z*(1-x[1]) - x[0],x[1])  -x[0]*Lambda_H + z*x[1]*Lambda_E - beta * y 
            bnds = ((b, cp.grid_max ),(0+1e-4,1- 1e-4))
            cons = ({'type': 'ineq', 'fun': lambda x:  R * a + w*z*(1-x[1])-b -x[0]}, {'type': 'ineq', 'fun': lambda x: x[0]})
            h0 = [b, .438]
            #print(h0)
            h_star = optimize.minimize(obj, h0, bounds = bnds,constraints=cons)
            #h_star3= fminbound(obj, b, R * a + w*z + b)
            #print(obj(h_star.x[0]), obj(h_star3))
            if h_star.success != True:
                h_star = optimize.minimize(obj, h0, bounds = bnds,constraints=cons, options={'eps': 1.4901161193847656e-02,  'maxiter': 100, 'ftol': 1e-05})
                if h_star.success != True:
                    print(h_star.message)
            #print(h_star.x[1],h_star.x[0])
            if h_star.x[1] == .4328:
                print(a)
            new_h[i_a, i_z],new_l[i_a, i_z], new_V[i_a, i_z] = h_star.x[0],h_star.x[1], -obj(h_star.x)
        if return_policy:
            return new_h[i_a,:], new_l[i_a, :]
        else:
            return new_V[i_a,:]

    rang = np.arange(len(asset_grid))
    Pool = ProcessingPool(96)
    new = Pool.map(do_bell, rang)
    #Pool.clear
    return np.asarray(new)



def initialize(cp):
    """
    Creates a suitable initial conditions V and c for value function and time
    iteration respectively.

    Parameters
    ----------
    cp : ConsumerProblem
        An instance of ConsumerProblem that stores primitives

    Returns
    -------
    V : array_like(float)
        Initial condition for value function iteration
    h : array_like(float)
        Initial condition for Coleman operator iteration

    """
    # === Simplify names, set up arrays === #
    R, beta, u, b = cp.R, cp.beta, cp.u, cp.b
    asset_grid, z_vals = cp.asset_grid, cp.z_vals
    shape = len(asset_grid), len(z_vals)
    V, h, c= np.empty(shape), np.empty(shape), np.empty(shape)

    # === Populate V and c === #
    for i_a, a in enumerate(asset_grid):
        for i_z, z in enumerate(z_vals):
            h_max = R * a + z + b
            h[i_a, i_z] = 0
            V[i_a, i_z] = u(h_max,.438) / (1 - beta)
            c[i_a, i_z] = h_max

    return V, h,c

@jit
def series(T, a, h_val, l_val,z_rlz, z_seq, z_vals, hf,lf):
    for t in range(T-1):
        i_z = z_seq[t] #this can probably be vectorized 
        a[t+1] = hf(a[t], i_z)
        h_val[t] = a[t+1] #this can probably be vectorized 
        l_val[t] = lf(a[t], i_z) #this can probably be vectorized 
        z_rlz[t]=z_vals[i_z]*(1-l_val[t]) #this can probably be vectorized 
   
    
    l_val[T-1] = lf(a[T-1], z_seq[T-1])
    h_val[T-1] = hf(a[T-1], z_seq[T-1])
    z_rlz[T-1] = z_vals[z_seq[T-1]]*(1-l_val[T-1]) 
    t1 = time.time()
    return a, h_val, l_val, z_rlz

def compute_asset_series_bell(cp,z_seq, T=T, verbose=False):
    """
    Simulates a time series of length T for assets, given optimal savings
    behavior.  Parameter cp is an instance of consumerProblem
    """

    Pi, z_vals, R, w= cp.Pi, cp.z_vals, cp.R, cp.w  # Simplify names
    #mc = MarkovChain(Pi)
    v_init, h_init, c_init = initialize(cp)
    K_bell = lambda c: bellman_operator(c, cp, return_policy=False)
    v = qe.compute_fixed_point(K_bell, v_init, verbose= False, max_iter =250, error_tol = tol_bell)
    policy = bellman_operator(v, cp, return_policy=True)
    h =  policy[:,0]
    l = policy[:,1]
    #vf = lambda a, i_z: np.interp(a, cp.asset_grid, v[:, i_z])
    #hf = lambda a, i_z: np.interp(a, cp.asset_grid, h[:, i_z])
    #lf = lambda a, i_z: np.interp(a, cp.asset_grid, l[:, i_z])
    asset_grid = cp.asset_grid
    k = cp.k
    @jit(nopython=True)
    def hf(a, i_z):
        i = int((a-asset_grid[0])/k)
        #return linterp(a, asset_grid[i:i+2], h[i:i+2, i_z])
        #x, xp, yp = a, asset_grid[i:i+2], h[i:i+2, i_z]
        #return (yp[0]*(xp[1]-x) + yp[1]*(x-xp[0]))/(xp[1]-xp[0])
        y = (h[i,i_z]*(asset_grid[i+1]-a) + h[i+1,i_z]*(a-asset_grid[i]))/(asset_grid[i+1]-asset_grid[i])
        return y
    
    @jit(nopython=True)
    def lf(a, i_z):
        i = int((a-asset_grid[0])/k)
        #return linterp(a, asset_grid[i:i+2], h[i:i+2, i_z])
        #x, xp, yp = a, asset_grid[i:i+2], h[i:i+2, i_z]
        #return (yp[0]*(xp[1]-x) + yp[1]*(x-xp[0]))/(xp[1]-xp[0])
        y = (l[i,i_z]*(asset_grid[i+1]-a) + l[i+1,i_z]*(a-asset_grid[i]))/(asset_grid[i+1]-asset_grid[i])
        return y

    a = np.zeros(T)
    a[0] = cp.b
    #z_seq = mc.simulate(T)
    z_rlz = np.zeros(T) #total labour supply after endogenous decisions. That is, e*(1-l)
    h_val = np.zeros(T)
    l_val = np.zeros(T) #liesure choice l! do NOT confuse with labour 
    a, h_val, l_val, z_rlz = series(T, a, h_val,l_val, z_rlz, z_seq, z_vals, hf, lf)
    
    return np.asarray(a), np.asarray(z_rlz), np.asarray(h_val), np.asarray(l_val), policy


@jit
def K_to_rw(K, L,fp):
    #ststkfp.AA=alpha/((1/beta-1)+delta)
    #fp.AA= 1.0/(ststkfp.AA**alpha *  L**(1.0-alpha)  )
    r = fp.AA*fp.alpha*np.power(K,fp.alpha-1)*np.power(L,1-fp.alpha)- fp.delta
    w = fp.AA*(1-fp.alpha)*np.power(K,fp.alpha)*np.power(L,-fp.alpha)
    fkk = fp.AA*(fp.alpha-1)*fp.alpha*np.power(K,fp.alpha-2)*np.power(L,1-fp.alpha)
    return r,w, fkk


def r_to_w(r,fp,cp):
    """
    Equilibrium wages associated with a given interest rate r.
    """
    delta = fp.delta 
    alpha = fp.alpha
    beta= cp.beta 
    #ststkfp.AA=fp.alpha/((1/cp.beta-1)+fp.delta)
    #fp.AA= 1.0/(ststkfp.AA**fp.alpha *  L**(1.0-fp.alpha)  )
    #A =1
    return fp.AA * (1 - fp.alpha) * (fp.AA * fp.alpha / (r + fp.delta))**(fp.alpha/ (1 - fp.alpha))

@jit
def r_to_K(r, L,fp):
    #ststkfp.AA=alpha/((1/beta-1)+delta)
    #fp.AA= 1.0/(ststkfp.AA**alpha *  L**(1.0-alpha)  )
    K = L*(((r +fp.delta)/(fp.AA*fp.alpha)))**(1/(fp.alpha-1))
    return K

def KL_to_Y(K,L,fp):

    return fp.AA*np.power(K,fp.alpha)*np.power(L,1-fp.alpha)

@jit
def coef_var(a):
    return np.sqrt(np.mean((a-np.mean(a))**2))/np.mean(a)



@jit
def compute_agg_prices(cp,z_seq, social= 0):
    a, z_rlz, h_val, l_val, policy = compute_asset_series_bell(cp, z_seq, T=T)
    agg_K = np.mean(a)
    L = np.mean(z_rlz)
    H = np.mean(1-l_val)
    coefvar = coef_var(a)
    r,w, fkk = K_to_rw(agg_K, L, fp)
    if social == 1:
        Lambda = cp.beta*agg_K*fkk*np.mean(\
                    cp.du(a*(1+r) + w*z_rlz - h_val)*((a/agg_K) - (z_rlz/L))\
                    )
    else:
        Lambda = 0
    return r, w, Lambda, agg_K, L, H, coefvar, a, z_rlz, h_val, l_val, policy 


def Gamma_IM(r):
    """
    Function whose zero is the Incomplete Markets allocation. 
    """
    #print('testing interst rate {}'.format(r))
    cp.r = r
    cp.R = 1+ cp.r
    cp.w = r_to_w(cp.r,fp, cp)
    r_nil, w_nil, Lambda_supply, K_supply, L_supply, Hours, coefvar, a, z_rlz, h_val, l_val, policy= compute_agg_prices(cp,z_seq, social= 0)
    K_demand = r_to_K(cp.r, L_supply,fp)
    excesssupply_K = K_supply- K_demand
    print ('With interest rate {}, excess capital supply is {}, hours are {}, labour supply is {}, k_supply is {}, k_demand is {}'.format(r*100, excesssupply_K, Hours,L_supply, K_supply, K_demand))

    return excesssupply_K

def firstbest(fp,cp):
    #ststkfp.AA=alpha/((1/beta-1)+delta)
    #fp.AA= 1.0/(ststkfp.AA**alpha *  L**(1.0-alpha)  )
    ##Get average productivity
    mc = MarkovChain(cp.Pi)
    stationary_distributions = mc.stationary_distributions[0]
    E = np.dot(stationary_distributions, cp.z_vals)
    def fbfoc(l):
        #l = np.min([np.max([x,0.001]),.999])
        L = E*(1-l)
        K = L*(((1-cp.beta +fp.delta*cp.beta)/(fp.AA*fp.alpha*cp.beta))**(1/(fp.alpha-1)))
        Y = fp.AA*(K**fp.alpha)*(L**(1-fp.alpha))
        Fl = -E*fp.AA*(1-fp.alpha)*(K**fp.alpha)*(L**(-fp.alpha))
        diff = cp.du(Y - fp.delta*K)*Fl + cp.dul(l)
        #print(cp.du(Y - fp.delta*K)*Fl )
        return diff
    
    l = fsolve(fbfoc, .5)[0]
    Labour_Supply = E*(1-l)
    L = E*(1-l)
    Hours = 1-l
    K = L*(((1-cp.beta +fp.delta*cp.beta)/(fp.AA*fp.alpha*cp.beta))**(1/(fp.alpha-1)))
    Y = fp.AA*np.power(K,fp.alpha)*np.power(L,1-fp.alpha)
    #r = fp.AA*fp.alpha*np.power(K,fp.alpha-1)*np.power(L,1-fp.alpha)- fp.delta
    r,w,fkk = K_to_rw(K, L,fp)
    return r, K, Hours,Labour_Supply, Y, w 



#######################################################
# Main routine
#######################################################

if __name__ == "__main__":

    #===Load model files and set tolerances==#

    cp = ConsumerProblem()
    fp = FirmProblem()
    fp.__init__(delta = .083)
    tol_brent = 10e-7
    tol_bell = 10e-4
    eta = 1
    cp.__init__(grid_max = 60, grid_size= 750, beta = .887)

    #===Load the Model Files===#

    model = open('pjmas2.mod', 'rb')

    model_in = pickle.load(model)

    name = model_in["filename"]

    cp.gamma_c = model_in["gamma_c"] 
    cp.gamma_l = model_in["gamma_l"]
    cp.A_L     = model_in["A_L"]
    cp.Pi      = np.asarray(model_in["Pi"])
    cp.z_vals  = np.asarray(model_in["z_vals"])

    model.close()

    #====Normalize mean of Labour distributuons===#

    mc = MarkovChain(cp.Pi)
    stationary_distributions = mc.stationary_distributions
    mean = np.dot(stationary_distributions, cp.z_vals)
    cp.z_vals = cp.z_vals/ mean   ## Standardise mean of avaible labour supply to 1
    z_seq = mc.simulate(T)


    #===Create Dict for Saving Results===#

    Results = {}

    Results["Name of Model"] = model_in["name"]

    #====Calcuate First Best===#

    fb_r, fb_K, fb_H, fb_L, fb_Y, fb_w  = firstbest(fp,cp)

    print('Runnung model {}, with grid_zize {}, max assets {}, T length of {}, prob matrix {}, z_vals {}, gamma_c {}, gamma_l{}'.format(name, len(cp.asset_grid), np.max(cp.asset_grid), T, cp.Pi, cp.z_vals, cp.gamma_c, cp.gamma_l))

    print('First best output {}, capital {}, interest rate {}, hours {} and labour supply {}'.format(fb_Y, fb_K, fb_r, fb_H, fb_L))

    results_FB = dict( (name, eval(name)) for name in ['fb_Y', 'fb_K', 'fb_r', 'fb_w', 'fb_H', 'fb_L'])


    #====Calcuate Incomplete Market Results ===#

    eqm_r_IM = brentq(Gamma_IM, -fp.delta*.95, (1-cp.beta)/cp.beta, xtol = tol_brent)


    im_r, im_w, im_Lambda, im_K, im_L, im_H, im_coefvar, im_a, im_z_rlz, im_h_val, im_l_val, im_policy  = compute_agg_prices(cp,z_seq, social =1)
    
    im_Y = KL_to_Y(im_K, im_L, fp)

    im_gini_a = gini(im_a)

    im_gini_i = gini(im_z_rlz)

    results_IM = dict( (name, eval(name)) for name in ['im_r', 'im_w', 'im_Lambda', 'im_K', 'im_L',\
                                                        'im_H', 'im_coefvar', 'im_a', 'im_z_rlz', 'im_h_val',\
                                                        'im_l_val', 'im_policy', 'im_Y', 'im_gini_a',  'im_gini_i'])
    

    print('Incomplete market capital {}, interest rate {}, hours {} and labour supply {} and Lambda {}'.format(im_K, im_r, im_H, im_L, im_Lambda))
  
    cp.r = eqm_r_IM
    cp.R = 1+cp.r
    cp.w = r_to_w(cp.r,fp, cp)
    cap_lab_ratio = ((cp.r+fp.delta)/fp.alpha)**(1/(fp.alpha-1))
    
    #=== Implement iteration Lambda(t+1) = Lambda(t-1) + (1-eta^t)(Lambda_t_hat- Lambda_t-1)==========#
    #-------------------------------------------------------------------------------------------------#
    #---- At t= 0, Lambda_t is taken as IM Lambda and Lambda_t-1 is 0
    #-----Lambda_t_hat is the Lambda implied by consumer asset demands and labour supply


    cp.Lambda_H =   im_Lambda
    cp.Lambda_E =   -(1-eta)*(im_Lambda*cap_lab_ratio/cp.beta)

    #---Dictionary to save iterations----$

    iterations = {}

    error1 = 1
    i= 0

    while np.abs(error1)> 4e-5 and i< 20 :

        eqm_r_IM = brentq(Gamma_IM, -fp.delta*.95, (1-cp.beta)/cp.beta , xtol = tol_brent)
        cp.r  = eqm_r_IM
        cp.R = 1+ cp.r
        cp.w = r_to_w(cp.r,fp, cp)
        CP_r,CP_w, CP_Lambda, CP_K, CP_L, CP_H, CP_coefvar, CP_a, CP_z_rlz, CP_h_val, CP_l_val, CP_policy = compute_agg_prices(cp,z_seq, social =1)

        Lambda = CP_Lambda 

        cap_lab_ratio = ((cp.r+fp.delta)/fp.alpha)**(1/(fp.alpha-1))
        
        error1 = cp.Lambda_H  - Lambda

        cp.Lambda_H = (1-eta)*cp.Lambda_H + eta*Lambda
        cp.Lambda_E = -(Lambda*cap_lab_ratio/cp.beta)*eta + (1-eta)*cp.Lambda_E


        print('Iteration {}, interest rate {}, capital stock {}, hours worked {}, labour suppl {}, Lambda {}, error1 {}'.format(i, CP_r, CP_K, CP_H, CP_L, CP_Lambda,error1))

        iterations[i] = [i, CP_r, CP_K, CP_H, CP_L, CP_Lambda, cp.Lambda_H, cp.Lambda_E, error1]

        i = i+1



CP_Y = KL_to_Y(CP_K, CP_L, fp)
CP_gini_a = gini(CP_a)

CP_gini_i = gini(CP_z_rlz)

results_CP = dict( (name, eval(name)) for name in ['CP_r' ,'CP_w', 'CP_Lambda', 'CP_K', 'CP_L', 'CP_H',\
                                                    'CP_coefvar', 'CP_a', 'CP_z_rlz', 'CP_h_val', 'CP_l_val',\
                                                    'CP_policy', 'CP_Y', 'CP_gini_i', 'CP_gini_a'])

Results["Results_FB"] = results_FB
Results["Results_IM"] = results_IM
Results["Results_CP"] = results_CP
Results["params"]     = [eta, T, tol_bell, tol_brent, cp.grid_max, cp.grid_size]
Results["iterations"]  = iterations 

Results["Con_In"]     = cp
Results["F_In"]       = fp


results_s = open('{}.res'.format(name), 'wb') 

pickle.dump(Results, results_s)

results_s.close()






