## setting 1 in simulation ##

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import time as tm

'''
Now we consider multivariate
t is now a d-dimensional vector
dim is the number of dimensions: len(t) = dim
'''
## 5/19 update theta function
def theta_t(t, n, k=0.05, c=0.01):
    # Generate theta(t)
    # Let dim be some number irrelevant to the structure
    # Take summations for all t[i], for example
    # theta = np.exp(0.01 * t * np.arange(1, n + 1))
#     theta = sum([np.exp(k * t[i]) for i in range(len(t))])* np.arange(1, n + 1) + c* np.arange(1, n + 1)
    theta = np.exp(k * np.sum(t)* np.arange(1, n + 1))
    # + c* np.arange(1, n + 1)
    theta -= np.mean(theta)
    return theta


'''
K is the kernel function. It takes two arguments: u and h.
u is the difference between the time of the comparison and the time of interest.
h is the bandwidth of the kernel.
The function returns the value of the kernel function at u, normalized by h.
'''

# Here u is a vector with dimension dim;
# Here we consider a formality of multiplication

def K(u, h):
    uh = u / h
    return np.prod(0.75 * (1 - uh ** 2) * (np.abs(uh) <= 1) / h)

'''
generate_data generates synthetic data for the model.
L_ij is the number of comparisons for each pair of entities.
p is the probability of edge existence.
n is the number of entities.
The function returns a dictionary with the following keys:
- edge: Adjacency matrix of the graph.
- Y_l: Binary outcomes of the comparisons.
- time: Time points of the comparisons.
'''

## 4/20 update: to ensure each model interacts with at least one other model
## 4/22 update: specifying k in theta_t
def generate_data(L_ij, p, n, dim, k, c):
    start_time = tm.time()  # Start timing using the time module

    L = max(L_ij)
    edge = np.full((n, n), np.nan)
    Y_l = np.full((n * n, L), np.nan)
    time_points = np.full((n * n, L, dim), np.nan)  # Renamed variable to avoid conflict

    for i in range(n - 1):
        for j in range(i + 1, n):
            edge[i, j] = np.random.binomial(1, p)
            edge[j, i] = edge[i, j]
            if edge[i, j] == 1:
                time_points[n * i + j, :L_ij[n * i + j], ] = np.random.uniform(0, 1, size=(L_ij[n * i + j], dim))
                time_points[n * j + i, :L_ij[n * i + j], ] = time_points[n * i + j, :L_ij[n * i + j], ]
                # Assuming theta_t function is defined elsewhere and provided
                theta = [theta_t(t, n, k, c) for t in time_points[n * i + j, :L_ij[n * i + j]]]
                for kk in range(L_ij[n * i + j]):
                    wj = np.exp(theta[kk][j])
                    wi = np.exp(theta[kk][i])
                    Y_l[n * i + j, kk] = np.random.binomial(1, wj / (wi + wj))  # j wins
                Y_l[n * j + i, :L_ij[n * i + j]] = 1 - Y_l[n * i + j, :L_ij[n * i + j]]
                
    for i in range(n - 1):
        if not np.any(edge[i, :] == 1):
            # Select a random index to set to 1
            j = np.random.choice([x for x in range(n) if x != i])  # Avoid setting the diagonal
            edge[i, j] = 1
            edge[j, i] = 1  # Ensure the matrix is symmetric
            time_points[n * i + j, :L_ij[n * i + j], ] = np.random.uniform(0, 1, size=(L_ij[n * i + j], dim))
            time_points[n * j + i, :L_ij[n * i + j], ] = time_points[n * i + j, :L_ij[n * i + j], ]
            # Assuming theta_t function is defined elsewhere and provided
            theta = [theta_t(t, n, k, c) for t in time_points[n * i + j, :L_ij[n * i + j]]]
            for kk in range(L_ij[n * i + j]):
                wj = np.exp(theta[kk][j])
                wi = np.exp(theta[kk][i])
                Y_l[n * i + j, kk] = np.random.binomial(1, wj / (wi + wj))  # j wins
            Y_l[n * j + i, :L_ij[n * i + j]] = 1 - Y_l[n * i + j, :L_ij[n * i + j]]


    print(f"generate_data runtime: {tm.time() - start_time} seconds")  # Print elapsed time using the time module
    return {'edge': edge, 'Y_l': Y_l, 'time': time_points}  # Updated key in the return dictionary


'''
L_t2 is the log-likelihood function for the model.
Y_l is the binary outcomes of the comparisons.
time is the time points of the comparisons.
edge is the adjacency matrix of the graph.
L_ij is the number of comparisons for each pair of entities.
theta is the vector of theta values.
t is the time of interest.
h is the bandwidth of the kernel.
lambda_val is the L2 penalization parameter.
n is the number of entities.
p is the probability of edge existence.
The function returns the value of the log-likelihood function at theta.
'''

## h is now 1D array with len(h) = dim
def L_t2(Y_l, time_points, edge, L_ij, theta, t, h, lambda_val, n, p):
    # start_time = tm.time() 
    L = max(L_ij)
    M = 0
    Z = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if edge[i, j] == 1:
                for k in range(L_ij[n * i + j]):
                    time_diff = time_points[n * i + j, k] - t
                    kernel_val = K(time_diff, h)
                    Z += kernel_val
                    M += kernel_val * (
                            -Y_l[n * i + j, k] * (theta[j] - theta[i]) +
                            np.log(1 + np.exp(theta[j] - theta[i]))
                    )
    regularization_term = lambda_val / 2 * np.sum(theta ** 2)
    ss = (n**2)*p*L
    # print(f"L_t2 runtime: {tm.time() - start_time} seconds")  # Print elapsed time
    
    return M / Z + regularization_term

'''
gradient2 is the gradient function for the model.
Y_l is the binary outcomes of the comparisons.
time is the time points of the comparisons.
edge is the adjacency matrix of the graph.
L_ij is the number of comparisons for each pair of entities.
theta is the vector of theta values.
t is the time of interest.
h is the bandwidth of the kernel.
lambda_val is the L2 penalization parameter.
n is the number of entities.
p is the probability of edge existence.
The function returns a dictionary with the following keys:
- grad1: The gradient of the log-likelihood function at theta.
- grad2: The Hessian of the log-likelihood function at theta.
'''
def gradient2(Y_l, time_points, edge, L_ij, theta, t, h, lambda_val, n, p):
    start_time = tm.time()  # Start timing using the time module correctly

    # Initialize grad1 as a zero vector of length n
    grad1 = np.zeros(n)
    # Initialize grad2 as a zero matrix of size n x n
    grad2 = np.zeros((n, n))
    # E is the identity matrix of size n x n
    E = np.eye(n)
    # Calculate Z, which is used in the normalization of grad1 and grad2
    Z = 0
    L = max(L_ij)
    
    # Iterate over all (i, j) pairs
    for i in range(1, n):  # Adjusted for Python's 0-indexing
        for j in range(i):
            if edge[j, i] == 1:
                time_diff_mat = time_points[n * i + j, :L_ij[n * i + j]] - t  # Use renamed parameter 'time_points'
                K_values = np.array([K(TD, h) for TD in time_diff_mat])
                # print(K_values)
                # Update the denominators
                Z_update = np.sum(K_values)
                # Update grad1 based on the formula provided
                grad1_update = np.sum(K_values * (-Y_l[n * i + j, :L_ij[n * i + j]] + np.exp(theta[j]) / (np.exp(theta[i]) + np.exp(theta[j])) )) * (E[j] - E[i])

                Z += Z_update
                grad1 += grad1_update

                # Update grad2 based on the formula provided
                grad2_update = np.sum(K_values) * np.exp(theta[i] + theta[j]) / (( np.exp(theta[i]) + np.exp(theta[j])) ** 2) * np.outer((E[i] - E[j]), (E[i] - E[j]).T)
                # grad2_update[j,j] = - np.sum(K_values) * np.exp(2*theta[j]) / (( np.exp(theta[i]) + np.exp(theta[j])) ** 2)
                grad2 += grad2_update

    ss = (n**2)*p*L
    # Normalize grad1 and grad2 by Z and add regularization term to grad1
    grad1 = grad1 / Z + lambda_val * theta
    grad2 = grad2 / Z

#     print(f"gradient2 runtime: {tm.time() - start_time} seconds")  # Print elapsed time using the time module

    return {'grad1': grad1, 'grad2': grad2}




'''
f_theta_h2 is the function that estimates the trajectory of theta(t).
Y_l is the binary outcomes of the comparisons.
time is the time points of the comparisons.
edge is the adjacency matrix of the graph.
L_ij is the number of comparisons for each pair of entities.
t is the time of interest.
lambda_val is the L2 penalization parameter.
h is the bandwidth of the kernel.
n is the number of entities.
p is the probability of edge existence.
The function returns the estimated trajectory of theta(t).
'''

# t=<300: 0.1; t>300: 0.1/(t-300) 
def f_theta_h2(Y_l, time_points, edge, L_ij, t, lambda_val, h, n, p, a=0.1, b=0.1, m=0.1):
    start_time = tm.time()
    theta_h = np.zeros(n)  # Assuming 'n' is defined elsewhere as the number of entities
    grad1 = gradient2(Y_l,time_points, edge, L_ij, theta_h, t, h, lambda_val, n, p)['grad1']
    sum_iter = 0
    ## This threshold is tunable
    ## For multivariate, this is likely all we could do
    ## Reduce the lim of iters; let it go!
    while np.sum(np.abs(grad1)) > 1e-6 and sum_iter <= 500:
    ## while np.sum(grad1 ** 2) > 1e-6 and sum_iter <= 50:
        m = 0.1 if m < 300 else 0.1 / (m - 299)
        sum_iter += 1
        reduce = 0
        while L_t2(Y_l, time_points, edge, L_ij, theta_h - m * grad1, t, h, lambda_val, n, p) > \
                L_t2(Y_l, time_points, edge, L_ij, theta_h, t, h, lambda_val, n, p) - a * m * np.dot(grad1, grad1):
            m = b * m
            reduce += 1
        theta_h = theta_h - m * grad1
        grad1 = gradient2(Y_l, time_points, edge, L_ij, theta_h, t, h, lambda_val, n, p)['grad1']
        if reduce >= 4:
            break
        # print(np.sum(np.abs(grad1)))
    # print(f"Iteration {sum_iter}, Step size {m}, Gradient sum {np.sum(np.abs(grad1))}")

    # print("sum_iter:", sum_iter)
    ## print("reduce:", reduce)
    # print(f"f_theta_h2 runtime: {tm.time() - start_time} seconds")  # Print elapsed time
    return theta_h

    
## Generate Statistic W
## T is the time grid
## theta_x is n*n*L, n -- theta_hat_i(X_im^l) 
## 4/21 update: choose pp% X_im in GMB
def generate_W(Y_l, time, edge, t, L_ij, h, n, p, w_iter, xi, theta_x, dim, idx):
    ## Suppose that t is the anchored parameter: t is an dim-dimensional iterable
    ## Replace m in the original code to be translated
    ## t is the real time, theta_h = theta_h(t) is the estimated value for plug-in
    start_time = tm.time()
    ## Initialization
    Vm = np.zeros((n, w_iter))
    Gm = np.zeros((n, w_iter))

    ## iter_1 for m
    ## iter_2 for k
    for iter_1 in range(n):
        for iter_2 in range(n):
            if  iter_1 != iter_2 and edge[iter_1, iter_2] == 1:
                t0 = time[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1]] - t
                K_values = np.array([K(TD, h) for TD in t0]) # L_ij*1
                idx0 = idx[n * iter_2 + iter_1,:]
                K_values[idx0==0] = 0
                ## Should be the same among groups
                Vm[iter_1,] += np.sum(K_values * (np.exp(theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_1] - theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_2]) /
                                                  (1 + np.exp(theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_1] - theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_2])) ** 2))
                for iter_samples in range(w_iter):
                        Gm[iter_1, iter_samples] += (K_values * xi[n * iter_2 + iter_1, : ,iter_samples]) @ \
                                            (-Y_l[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1]] +
                                             np.exp(theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_1]) / (np.exp(theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_1]) + np.exp(theta_x[n * iter_2 + iter_1,:L_ij[n * iter_2 + iter_1],iter_2])))

    W = ((np.sqrt(h)) ** dim) * np.divide(Gm, Vm, where = Vm != 0, out = np.zeros_like(Gm))
    ## W = - np.sqrt(n * p * max(L_ij) * h) * (Gm / Vm)
    print(f"generate W runtime: {tm.time() - start_time} seconds")  # Print elapsed time
    return W


# Now `estimates` contains the theta estimates for each simulation
# Subject to edits in multivariate
def plot_trajectory(estimates, true_theta, t):
    """
    Plots the estimated trajectories.

    Parameters:
    - estimates: Array of estimated theta values.
    - t: Time points.
    - true_theta: Array of true theta values for comparison (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, estimates[:, 40], label=f'Estimate {1} for node 40')

    if true_theta is not None:
        plt.plot(t, true_theta[:, 40], 'k--', label='True Theta')

    plt.xlabel('Time')
    plt.ylabel('Theta')
    plt.title('Trajectory Estimates')
    plt.legend()
    plt.show()


def parse_argument():
    parser = argparse.ArgumentParser(description='Estimate the trajectory of theta(t)')
    parser.add_argument('--h', type=float, default=0.3, help='Kernel bandwidth')
    parser.add_argument('--time_num', type=int, default=100, help='Number of time points')
    parser.add_argument('--lambda_val', type=float, default=0.00001, help='L2 penalization')
    parser.add_argument('--n', type=int, default=100, help='Number of entities')
    parser.add_argument('--p', type=float, default=0.2, help='Probability for edge existence')
    parser.add_argument('--L_val', type=int, default=400, help='Number of comparisons for each pair of entities')
    parser.add_argument('--iter_num', type=int, default=50, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', type=int, default=8, help='Number of CPUs')
    # Add the default number dim
    parser.add_argument('--dim', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--w_iter', type=int, default=100, help='Samples for Iteration for Confidence Band')
    parser.add_argument('--k', type=float, default=0.05, help='Size of theta')
    parser.add_argument('--m', type=float, default=0.1, help='Learning rate in GD')
    parser.add_argument('--pp', type=float, default=400, help='Number of samples in GMB')
    parser.add_argument('--c', type=float, default=0.1, help='Size of the intecept in theta')
    args = parser.parse_args()
    print(args)
    return args

## Make this a function of dim as well
## Supposedly, t is now a dim-dimensional grid
def compute_theta_x(Y_l, time, edge, L_ij, t, lambda_val, h, n, p, m, k, c):  # Added k if needed
    start_time = tm.time()
    # Estimation for theta(t) and theta_x
    if np.isnan(t).all():
        theta_x = np.full(n, 0)
        ground_truth = np.full(n, 0)
    else:
        theta_x = f_theta_h2(Y_l, time, edge, L_ij, t, lambda_val, h, n, p, m)
        ground_truth = theta_t(t, n, k, c)  # Ensure k is defined or passed
    print(f"theta_x runtime: {tm.time() - start_time} seconds")  # Print elapsed time
    return t, theta_x, ground_truth

def compute_theta_h(Y_l, time, edge, L_ij, t, lambda_val, h, n, p, m, w_iter, dim, k, xi, theta_x, idx, c):
    start_time = tm.time()
    # Estimation for theta(t)
#     print(t)
    theta_h = f_theta_h2(Y_l, time, edge, L_ij, t, lambda_val, h, n, p, m)
    
    # Generate W matrix
    W = generate_W(Y_l, time, edge, t, L_ij, h, n, p, w_iter, xi, theta_x, dim, idx)
    
    # Generate ground truth
    ground_truth = theta_t(t, n, k, c) 
    print(ground_truth)
    # Return relevant values
    print(f"theta_h runtime: {tm.time() - start_time} seconds")  # Print elapsed time
    return t, theta_h, ground_truth, W


## Generate multivariate grids
def generate_grid(begin, end, dim, time_num):
    # Compute the number of points per dimension
    points_per_dim = int(round(time_num ** (1/dim)))

    # Generate a linearly spaced grid for each dimension
    grid1d = np.linspace(begin, end, points_per_dim)
    
    # Generate meshgrid for all dimensions
    mesh = np.meshgrid(*([grid1d]*dim))
    
    # Flatten and combine to form the final grid
    grid_points = np.vstack(map(np.ravel, mesh)).T
    
    # If the number of points exceeds time_num, truncate the array
    if len(grid_points) > time_num:
        grid_points = grid_points[:time_num]

    return grid_points


if __name__ == "__main__":
    args = parse_argument()
    h = args.h
    time_num = args.time_num
    lambda_val = args.lambda_val
    n = args.n
    L_val = args.L_val
    p = args.p
    iter_num = args.iter_num
    seed = args.seed
    w_iter = args.w_iter
    dim = args.dim
    k = args.k
    m = args.m
    pp = args.pp
    c = args.c
    
    total_start_time = tm.time()
    ## Let's try this for multivariate
    ## T = np.linspace(0, 1, time_num)
    T = generate_grid(0, 1, dim, time_num)
    time_num_new = len(T)
    ## I doubt if this is the real coef here
    ## This could be a typo
    L_ij = np.full((n * n,), L_val)
    
    np.random.seed(seed)
    ## Gaussian Multiplier Bootstrapping
    shape = (n * n, max(L_ij), w_iter)
    xi = np.random.normal(loc=0, scale=1, size=shape)
    data = generate_data(L_ij, p, n, dim, k, c)  # Generate synthetic data
    edge = data['edge']
    Y_l = data['Y_l']
    time0 = data['time']
    
    time_2d = time0.reshape(time0.shape[0] * time0.shape[1], time0.shape[2])
    
    ##### randomly choose Xim from time_2d
#     pp = 0.3
    time_2d_filt = np.full((time_2d.shape[0],dim), np.nan)

    # identify non-nan row in time_2d
    non_nan_rows = ~np.isnan(time_2d).any(axis=1)
    non_nan_row_indices = np.where(non_nan_rows)[0]
#     print(non_nan_row_indices)

    # choose pp%
    nn = non_nan_row_indices.shape[0]
    rr = pp/nn
    if rr<1:
        idx = np.random.binomial(1, rr,size = non_nan_row_indices.shape[0])
    else: 
        idx = np.random.binomial(1, 1,size = non_nan_row_indices.shape[0])
    
    non_nan_row_indices_filt = non_nan_row_indices[idx==1]
#     print(non_nan_row_indices_filt)

    idd = np.zeros(time_2d.shape[0])
    idd[non_nan_row_indices_filt] = 1
    idd0 = idd.reshape([n*n,L_val])
#     print(idd)
    time_2d_filt[non_nan_row_indices_filt,:] = time_2d[non_nan_row_indices_filt,:]

    ####### first calculate \hat{\theta_i}(\bX_im^l)
    
    all_combinations = [(Y_l, time0, edge, L_ij, t, lambda_val, h, n, p, m, k, c) for t in time_2d_filt]

    # Prepare for parallel processing
    start_time_parallel = tm.time()
    with ProcessPoolExecutor(max_workers=args.cpu) as executor:
        # Associate each future with its index in time_2d
        futures = {executor.submit(compute_theta_x, *comb): i for i, comb in enumerate(all_combinations)}

        results = [None] * len(all_combinations)  # Preallocate the results list

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task"):
            index = futures[future]  # Get the original index
            try:
                res = future.result()
                results[index] = res  # Place the result at the correct index
            except Exception as e:
                # Handle exceptions that might be raised during execution of the task
                print(f"Task at index {index} raised an exception: {e}")


    # Now, you can be sure that results[i] corresponds to all_combinations[i]

    # Organize results into a structured format
    t_x = np.zeros((n*n*L_val, dim))
    estimates_x = np.zeros((n*n*L_val, n))
    ground_truth_x = np.zeros((n*n*L_val, n))

    for ii in range(time_2d.shape[0]):
        res0 = results[ii]
        t_x[ii] = res0[0]
        estimates_x[ii] = res0[1]
        ground_truth_x[ii] = res0[2]

    # reshape theta_x 
    theta_x = estimates_x.reshape([n*n,L_val,n])

    
    ####### then calculate \hat{\theta_i}(\xb)
    all_combinations = [(Y_l, time0, edge, L_ij, t, lambda_val, h, n, p, m, w_iter, dim, k, xi, theta_x, idd0, c) for t in T]
    # Prepare for parallel processing
    start_time_parallel = tm.time()

    with ProcessPoolExecutor(max_workers=args.cpu) as executor:
        # Submit all the tasks and create a list of futures
        futures = {executor.submit(compute_theta_h, *comb): i for i, comb in enumerate(all_combinations)}

        results = [None] * len(all_combinations)  # Preallocate the results list

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="task"):
            index = futures[future]  # Get the original index
            try:
                res = future.result()
                results[index] = res  # Place the result at the correct index
            except Exception as e:
                # Handle exceptions that might be raised during execution of the task
                print(f"Task at index {index} raised an exception: {e}")


    print(f"Parallel processing runtime2: {tm.time() - start_time_parallel} seconds")

    # Organize results into a structured format
    estimates = np.zeros((time_num_new, n))
    ground_truth = np.zeros((time_num_new, n))
    # W is n-by-w_iter matrix among i in iter_num and t in time_num
    GMB_W = np.zeros((time_num_new, n, w_iter))

    for ii in range(time_num_new):
        res0 = results[ii]
    #     t_all[ii] = res0[0]
        estimates[ii] = res0[1]
        ground_truth[ii] = res0[2]
        GMB_W[ii] = res0[3]

#     # Organize results into a structured format
#     estimates = np.zeros((time_num, n))
#     ground_truth = np.zeros((time_num, n))
#     # W is n-by-w_iter matrix among i in iter_num and t in time_num
#     GMB_W = np.zeros((time_num, n, w_iter))

#     for t, theta_h, true_theta, W in results:
#         estimates[np.where(T == t)[0][0]] = theta_h
#         ground_truth[np.where(T == t)[0][0]] = true_theta
#         GMB_W[np.where(T == t)[0][0]] = W
        
    # Wrap up and print total runtime
    print(f"Total script runtime: {tm.time() - total_start_time} seconds")
    
    result_dict = {'estimates': estimates, 'ground_truth': ground_truth, 'GMB_W': GMB_W}
    np.savez(f'results/n={n}/n{n}_Lij{L_val}_p{p}_h{h}_lambda{lambda_val}_dim{dim}_m{m}_k{k}+{c}_num{pp}_seed{seed}_c10.npz', **result_dict)