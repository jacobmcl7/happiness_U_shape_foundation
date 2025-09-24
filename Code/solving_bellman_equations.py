# Bellman equation solution

#################################
# CONTENTS
#################################
# 1. Value function iteration (VFI) for a simple deterministic one-state, one-control problem
# 2. Backward induction for a time-dependent version of the same problem (with finite horizon)
#################################

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#################################
#### 1)
#################################

#### this first part solves the Bellman equation by value function iteration, in the case where there is one state, one control and no explicit time-dependence

# VFI for the following problem: max sum beta^t h(K_t,I_t) s.t. LoM K_{t+1} = f(K_t,I_t)
# Then simulates from K0 and plots K_t, I_t, h_t to show U-shape.

# Parameters
A = 1.0  # efficiency parameter
alpha = 0.33  # diminishing returns parameter
beta = 0.98  # discount factor
delta = 0.05  # depreciation rate
gamma = 1 # investment cost parameter

# grids
Kmin, Kmax, nK = 0.1, 8.0, 300
K_grid = np.linspace(Kmin, Kmax, nK)

Imin, Imax, nI = 0.0, 2.5, 201  # restrict investments to nonnegative for this example
I_grid = np.linspace(Imin, Imax, nI)

#### HERE IS WHERE WE DEFINE THE CONTEMPORANEOUS OBJECTIVE h(K,I) ####

# define the payoff function
def h_of(K, I):
    return A*K**alpha - gamma*I**2

#### ####

# Precompute h(K,I) on grid of possible (K, I) pairs, to save time in VFI loop
h_table = np.empty((nK, nI))
for iK, K in enumerate(K_grid):
    for iI, I in enumerate(I_grid):
        h_table[iK, iI] = h_of(K, I)


# Value function iteration
V = np.zeros(nK)  # initial guess
policy_I_idx = np.zeros(nK, dtype=int)  # this will store the index of the maximising I in I_grid for each K
tol = 1e-6
max_iter = 500

# we now repeatedly apply the Bellman operator: due to the CMT theory the resulting value and policy function should converge to the solution
for it in range(max_iter):

    # create vectors to hold the new value function and policy function index
    Vnew = np.empty_like(V)
    new_policy = np.empty_like(policy_I_idx)

    # create a linear interpolation of the current value function V for next-period K
    V_interp = interp1d(K_grid, V, kind='linear', fill_value='extrapolate', assume_sorted=True)

    # now loop over each inner K grid point, to find the best I

    for iK, K in enumerate(K_grid):

        #### HERE IS WHERE WE DEFINE THE LAW OF MOTION (e.g. Knext = (1-delta)*K + I) ####
        
        # for each possible I, compute Knext and value (i.e. we compute the value associated with choosing every I in the grid)
        Knext = (1-delta)*K + I_grid  # vector over I_grid giving the next period K

        #### ####

        # enforce Knext within interpolation domain by clipping
        Knext_clipped = np.clip(Knext, Kmin, Kmax)

        # evaluate the continuation value V at each value of Knext
        cont_val = V_interp(Knext_clipped)

        # calculate total value for each I choice, where we find the value of h and the continuation value for every I
        total = h_table[iK, :] + beta*cont_val

        # pick the best I
        jstar = np.argmax(total)

        # set the new value and policy associated with this optimal I
        Vnew[iK] = total[jstar]
        new_policy[iK] = jstar

    # compute the sup-norm distance between V and Vnew, to check for convergence
    diff = np.max(np.abs(Vnew - V))

    # update the value function and policy function
    V = Vnew
    policy_I_idx = new_policy

    # stop if converged
    if diff < tol:
        print(f'VFI converged in {it+1} iterations, diff={diff:.2e}')
        break

    # print some info every 50 iterations
    if it%50 == 0:
        print(f'VFI iteration {it}, diff={diff:.2e}')


# Construct policy function for I(K) after having found the indices
policy_I = I_grid[policy_I_idx]
policy_fun = interp1d(K_grid, policy_I, kind='linear', fill_value=(Imin, Imax), bounds_error=False)



# we have now solved the problem generally, so we can simulate the model from a given initial capital level K0 to get the dynamics

#### HERE IS WHERE WE SET INITIAL STOCK SIZE ####
K0 = 1.5  # note this must be in [Kmin, Kmax]
#### ####
Tsim = 80   # number of periods to simulate
Ks = np.empty(Tsim+1)   # these three are to store the values at time s
Is = np.empty(Tsim)
hs = np.empty(Tsim)
Ks[0] = K0
# now simulate forward, using the policy function to get I_t from K_t
for t in range(Tsim):
    I_t = float(policy_fun(Ks[t]))
    Is[t] = I_t
    hs[t] = h_of(Ks[t], I_t)
    Ks[t+1] = (1-delta)*Ks[t] + I_t
# last period h for final K (no I chosen beyond Tsim-1)



# Plot results: three separate plots (each its own figure)
plt.figure(figsize=(8,4))
plt.plot(np.arange(Tsim+1), Ks, marker='o', markersize=3)
plt.title('K_t (capital)')
plt.xlabel('time t')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(np.arange(Tsim), Is, marker='o', markersize=3)
plt.title('I_t (investment policy)')
plt.xlabel('time t')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(np.arange(Tsim), hs, marker='o', markersize=3)
plt.title('h_t = h(K_t,I_t) (period payoff)')
plt.xlabel('time t')
plt.grid(True)
plt.tight_layout()
plt.show()

# Also print first 12 values to inspect U-shape
print('t, K_t, I_t, h_t (first 12 periods)')
for t in range(12):
    print(f'{t:2d}, {Ks[t]:6.3f}, {Is[t]:6.3f}, {hs[t]:7.4f}')





#################################
#### 2)
#################################

#### this second part solves a time-dependent version of the same problem, using backward induction on a finite horizon

# maybe check more - it gives qualitatively similar results to the infinite-horizon case above when parameters are fixed, with some slightly different behaviour near the final period (to be expected), but worth checking

# define a function that does this all

def solve_and_simulate(K0, Tsim, gamma_path, delta_path, A, alpha, beta, h_of_t, K_grid, I_grid, K_next):

    #### DO BACKWARD INDUCTION TO SOLVE THE TIME-DEPENDENT PROBLEM

    # V[t, iK] holds value at time t for K_grid[iK]; V[Tsim,:] is terminal value = 0
    V = np.zeros((Tsim + 1, nK))
    policy_I_idx = np.zeros((Tsim, nK), dtype=int)  # index of optimal I for each (t, K_index)

    # Backwards loop
    # note that we keep V[Tsim, :] = 0 as the terminal condition: this is the transversality condition (TVC)
    for t in range(Tsim - 1, -1, -1):  # t = Tsim-1, ..., 0
        # create interpolator for V[t+1, .]
        Vnext_interp = interp1d(K_grid, V[t + 1], kind='linear', fill_value='extrapolate', assume_sorted=True)
        delta_t = delta_path[t]
        # loop across K grid
        for iK, K in enumerate(K_grid):

            # compute feasible K' for each candidate I (vector)
            Knext = K_next(K, I_grid, delta_t)

            # clip onto interpolation domain (alternatively expand K_grid to avoid clipping)
            Knext_clipped = np.clip(Knext, Kmin, Kmax)

            # calculate continuation value for each candidate I
            cont_val = Vnext_interp(Knext_clipped)        # V[t+1](K') vector of length nI

            # immediate payoff vector for all candidate I (length nI)
            h_vec = h_of_t(K, I_grid, t)
            total = h_vec + beta * cont_val               # total value for each I
            jstar = int(np.argmax(total))                 # index of best I
            V[t, iK] = total[jstar]
            policy_I_idx[t, iK] = jstar

        # Optionally print progress
        if (t % 10) == 0:
            print(f'Backward step done: t={t}')

    print('Backward induction finished.')

    # -------------------------
    # Build time-dependent policy function objects for simulation
    # -------------------------
    policy_I_vals = I_grid[policy_I_idx]   # shape (Tsim, nK)
    # note this is equivalent to policy_I_vals[t, iK] = I_grid[policy_I_idx[t, iK]]

    # build interpolator for each time period (so we can evaluate I*_t(K) at off-grid K)
    policy_fun = [
        interp1d(K_grid, policy_I_vals[t], kind='linear',
                fill_value=(Imin, Imax), bounds_error=False, assume_sorted=True)
        for t in range(Tsim)
    ]


    #### NOW SIMULATE

    Ks = np.empty(Tsim + 1)
    Is = np.empty(Tsim)
    hs = np.empty(Tsim)
    Ks[0] = K0

    for t in range(Tsim):
        # evaluate policy at current K_t using the t-th policy function
        I_t = float(policy_fun[t](Ks[t]))
        Is[t] = I_t
        hs[t] = (A * (Ks[t]**alpha) - gamma_path[t] * (I_t**2))   # h_t = h(K_t,I_t,t)
        # update K
        Ks[t + 1] = (1.0 - delta_path[t]) * Ks[t] + I_t


    #### NOW PLOT RESULTS

    plt.figure(figsize=(8,4))
    plt.plot(np.arange(Tsim+1), Ks, marker='o', markersize=3)
    plt.title('K_t (capital)')
    plt.xlabel('time t')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(np.arange(Tsim), Is, marker='o', markersize=3)
    plt.title('I_t (investment policy)')
    plt.xlabel('time t')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(np.arange(Tsim), hs, marker='o', markersize=3)
    plt.title('h_t = h(K_t,I_t) (period payoff)')
    plt.xlabel('time t')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print first 12 periods for inspection
    print('t, K_t, I_t, delta_t, gamma_t, h_t (first 12 periods)')
    for t in range(min(12, Tsim)):
        print(f'{t:2d}, {Ks[t]:6.3f}, {Is[t]:6.3f}, {delta_path[t]:.3f}, {gamma_path[t]:.3f}, {hs[t]:7.4f}')





# now use it


A = 1.0
alpha = 0.33
beta = 0.98
Tsim = 80  # number of decision periods (0..Tsim-1). V[Tsim] is terminal value (set to 0, by the transversality condition under the condition we need capital stock positive at final period).
gamma_path = np.linspace(0.5, 1, Tsim)**2  # increasing investment cost over time
delta_path = np.linspace(0.05, 0.05, Tsim)  # increasing depreciation over time
K0 = 1.5  # choose initial capital within [Kmin, Kmax]

# set the state and control grids as before
Kmin, Kmax, nK = 0.1, 8.0, 300
K_grid = np.linspace(Kmin, Kmax, nK)
Imin, Imax, nI = 0.0, 2.5, 201
I_grid = np.linspace(Imin, Imax, nI)

# precalculate payoff function depending on time t, h(K,I,t), vectorised in I
def h_of_t(K, I_grid, t):
    gamma_t = gamma_path[t]
    return A * (K**alpha) - gamma_t * (I_grid**2)

# define the law of motion function, dependent on time t
def K_next(K, I_grid, delta_t):
    return (1.0 - delta_t) * K + I_grid

solve_and_simulate(K0, Tsim, gamma_path, delta_path, A, alpha, beta, h_of_t, K_grid, I_grid, K_next)


























#################################################################
#################################################################
# ARCHIVE
#################################################################
#################################################################

# before making the function, part 2 looked like this:


A = 1.0
alpha = 0.33
beta = 0.98
Tsim = 80  # number of decision periods (0..Tsim-1). V[Tsim] is terminal value (set to 0, by the transversality condition under the condition we need capital stock positive at final period).
gamma_path = np.linspace(0.5, 1, Tsim)**2  # increasing investment cost over time
delta_path = np.linspace(0.05, 0.2, Tsim)  # increasing depreciation over time

# set the state and control grids as before
Kmin, Kmax, nK = 0.1, 8.0, 300
K_grid = np.linspace(Kmin, Kmax, nK)
Imin, Imax, nI = 0.0, 2.5, 201
I_grid = np.linspace(Imin, Imax, nI)

#### THIS IS WHERE WE DEFINE THE TIME-DEPENDENT CONTEMPORANEOUS OBJECTIVE h(K,I,t) ####

# precalculate payoff function depending on time t, h(K,I,t), vectorised in I
def h_of_t(K, I_grid, t):
    gamma_t = gamma_path[t]
    return A * (K**alpha) - gamma_t * (I_grid**2)

#### ####

# -------------------------
# Backward induction (finite-horizon DP)
# -------------------------
# V[t, iK] holds value at time t for K_grid[iK]; V[Tsim,:] is terminal value = 0
V = np.zeros((Tsim + 1, nK))
policy_I_idx = np.zeros((Tsim, nK), dtype=int)  # index of optimal I for each (t, K_index)

# Backwards loop
# note that we keep V[Tsim, :] = 0 as the terminal condition: this is the transversality condition (TVC)
for t in range(Tsim - 1, -1, -1):  # t = Tsim-1, ..., 0
    # create interpolator for V[t+1, .]
    Vnext_interp = interp1d(K_grid, V[t + 1], kind='linear', fill_value='extrapolate', assume_sorted=True)
    delta_t = delta_path[t]
    # loop across K grid
    for iK, K in enumerate(K_grid):

        #### THIS IS WHERE WE DEFINE THE TIME-DEPENDENT LAW OF MOTION ####

        # compute feasible K' for each candidate I (vector)
        Knext = (1.0 - delta_t) * K + I_grid

        #### ####

        # clip onto interpolation domain (alternatively expand K_grid to avoid clipping)
        Knext_clipped = np.clip(Knext, Kmin, Kmax)

        # calculate continuation value for each candidate I
        cont_val = Vnext_interp(Knext_clipped)        # V[t+1](K') vector of length nI

        # immediate payoff vector for all candidate I (length nI)
        h_vec = h_of_t(K, I_grid, t)
        total = h_vec + beta * cont_val               # total value for each I
        jstar = int(np.argmax(total))                 # index of best I
        V[t, iK] = total[jstar]
        policy_I_idx[t, iK] = jstar

    # Optionally print progress
    if (t % 10) == 0:
        print(f'Backward step done: t={t}')

print('Backward induction finished.')

# -------------------------
# Build time-dependent policy function objects for simulation
# -------------------------
policy_I_vals = I_grid[policy_I_idx]   # shape (Tsim, nK)
# note this is equivalent to policy_I_vals[t, iK] = I_grid[policy_I_idx[t, iK]]

# build interpolator for each time period (so we can evaluate I*_t(K) at off-grid K)
policy_fun = [
    interp1d(K_grid, policy_I_vals[t], kind='linear',
             fill_value=(Imin, Imax), bounds_error=False, assume_sorted=True)
    for t in range(Tsim)
]

# -------------------------
# Simulate forward from K0 using the time-dependent policy
# -------------------------
K0 = 1.5  # choose initial capital within [Kmin, Kmax]
Ks = np.empty(Tsim + 1)
Is = np.empty(Tsim)
hs = np.empty(Tsim)
Ks[0] = K0

for t in range(Tsim):
    # evaluate policy at current K_t using the t-th policy function
    I_t = float(policy_fun[t](Ks[t]))
    Is[t] = I_t
    hs[t] = (A * (Ks[t]**alpha) - gamma_path[t] * (I_t**2))   # h_t = h(K_t,I_t,t)
    # update K
    Ks[t + 1] = (1.0 - delta_path[t]) * Ks[t] + I_t

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(8,4))
plt.plot(np.arange(Tsim+1), Ks, marker='o', markersize=3)
plt.title('K_t (capital)')
plt.xlabel('time t')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(np.arange(Tsim), Is, marker='o', markersize=3)
plt.title('I_t (investment policy)')
plt.xlabel('time t')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(np.arange(Tsim), hs, marker='o', markersize=3)
plt.title('h_t = h(K_t,I_t) (period payoff)')
plt.xlabel('time t')
plt.grid(True)
plt.tight_layout()
plt.show()

# print first 12 periods for inspection
print('t, K_t, I_t, delta_t, gamma_t, h_t (first 12 periods)')
for t in range(min(12, Tsim)):
    print(f'{t:2d}, {Ks[t]:6.3f}, {Is[t]:6.3f}, {delta_path[t]:.3f}, {gamma_path[t]:.3f}, {hs[t]:7.4f}')



