import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import time as tm
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def K(u, h):
    uh = u / h
    return np.prod(0.75 * (1 - uh ** 2) * (np.abs(uh) <= 1) / h)

def L_t2(Y_l, time_points, edge, L_ij, theta, t, h, lambda_val, n, p):
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
    return M / Z + regularization_term

def gradient2(Y_l, time_points, edge, L_ij, theta, t, h, lambda_val, n, p):
    grad1 = np.zeros(n)
    grad2 = np.zeros((n, n))
    E = np.eye(n)
    Z = 0

    for i in range(1, n):
        for j in range(i):
            if edge[j, i] == 1:
                time_diff_mat = time_points[n * i + j, :L_ij[n * i + j]] - t
                K_values = np.array([K(TD, h) for TD in time_diff_mat])
                Z_update = np.sum(K_values)
                grad1_update = np.sum(K_values * (-Y_l[n * i + j, :L_ij[n * i + j]] +
                                                  np.exp(theta[j]) / (np.exp(theta[i]) + np.exp(theta[j])))) * (E[j] - E[i])
                Z += Z_update
                grad1 += grad1_update
                grad2_update = np.sum(K_values) * np.exp(theta[i] + theta[j]) / (
                            np.exp(theta[i]) + np.exp(theta[j])) ** 2 * np.outer((E[i] - E[j]), (E[i] - E[j]).T)
                grad2 += grad2_update

    grad1 = grad1 / Z + lambda_val * theta
    grad2 = grad2 / Z
    return {'grad1': grad1, 'grad2': grad2}

def f_theta_h2(Y_l, time_points, edge, L_ij, t, lambda_val, h, n, p, a=0.1, b=0.1, m=0.1):
    theta_h = np.zeros(n)
    grad1 = gradient2(Y_l, time_points, edge, L_ij, theta_h, t, h, lambda_val, n, p)['grad1']
    sum_iter = 0

    while np.sum(np.abs(grad1)) > 1e-6 and sum_iter <= 200:
        m = m
        sum_iter += 1
        reduce = 0
        while L_t2(Y_l, time_points, edge, L_ij, theta_h - m * grad1, t, h, lambda_val, n, p) > \
                L_t2(Y_l, time_points, edge, L_ij, theta_h, t, h, lambda_val, n, p) - a * m * np.dot(grad1, grad1):
            m = b * m
            reduce += 1
        theta_h = theta_h - m * grad1
        grad1 = gradient2(Y_l, time_points, edge, L_ij, theta_h, t, h, lambda_val, n, p)['grad1']
        if reduce >= 2:
            break
    return theta_h

def generate_W(Y_l, time_points, edge, t, L_ij, h, n, p, w_iter, xi, theta_x, dim, idx):
    Vm = np.zeros((n, w_iter))
    Gm = np.zeros((n, w_iter))

    for iter_1 in range(n):
        for iter_2 in range(n):
            if iter_1 != iter_2 and edge[iter_1, iter_2] == 1:
                t0 = time_points[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1]] - t
                K_values = np.array([K(TD, h) for TD in t0])
                idx0 = idx[n * iter_2 + iter_1, :]
                K_values[idx0 == 0] = 0
                Vm[iter_1,] += np.sum(K_values * (np.exp(theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_1] -
                                                  theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_2]) /
                                                  (1 + np.exp(theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_1] -
                                                              theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_2])) ** 2))
                for iter_samples in range(w_iter):
                    Gm[iter_1, iter_samples] += (K_values * xi[n * iter_2 + iter_1, :, iter_samples]) @ \
                                                (-Y_l[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1]] +
                                                 np.exp(theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_1]) / (
                                                             np.exp(theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_1]) +
                                                             np.exp(theta_x[n * iter_2 + iter_1, :L_ij[n * iter_2 + iter_1], iter_2])))

    W = ((np.sqrt(h)) ** dim) * np.divide(Gm, Vm, where=Vm != 0, out=np.zeros_like(Gm))
    return W

def compute_theta_x(Y_l, time_points, edge, L_ij, t, lambda_val, h, n, p, m):  # Added k if needed
    start_time = tm.time()
    if np.isnan(t).all():
        theta_x = np.full(n, 0)
    else:
        theta_x = f_theta_h2(Y_l, time_points, edge, L_ij, t, lambda_val, h, n, p, m)
    print(f"theta_x runtime: {tm.time() - start_time} seconds")
    return t, theta_x

def compute_theta_h(Y_l, time_points, edge, L_ij, t, lambda_val, h, n, p, m, w_iter, dim, xi, theta_x, idx):
    start_time = tm.time()
    theta_h = f_theta_h2(Y_l, time_points, edge, L_ij, t, lambda_val, h, n, p, m)
    W = generate_W(Y_l, time_points, edge, t, L_ij, h, n, p, w_iter, xi, theta_x, dim, idx)
    print(f"theta_h runtime: {tm.time() - start_time} seconds")
    return t, theta_h, W

# embedded t 
subs = ['anatomy', 'clinical_knowledge', 'medical_genetics', 'college_biology']
# Subject index 0-3: first command-line argument, else the SLURM array task id.
import sys
i = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
csv_path = f'data/prompt/embedded_questions_{subs[i]}.csv'
df = pd.read_csv(csv_path)

# Function to convert a string representation of a list to a NumPy array
def string_to_array(s):
    return np.fromstring(s.strip('[]'), sep=',')

# Apply the conversion function to each row in the vector column
embedding_column = df.iloc[:, 2]
embeddings = np.array(embedding_column.apply(string_to_array).tolist())
print(embeddings.shape)

test_csv_path = f'data/prompt/embedded_questions_test_{subs[i]}.csv'
df = pd.read_csv(test_csv_path)

# Apply the conversion function to each row in the vector column
embedding_column = df.iloc[:, 2]
test_embeddings = np.array(embedding_column.apply(string_to_array).tolist())
print(test_embeddings.shape)

embeddings0 = np.concatenate((embeddings,test_embeddings), axis=0)
print(embeddings0.shape)

desired_dim = 16  # Specify the desired dimensionality
pca = PCA(n_components=desired_dim)
reduced_embeddings0 = pca.fit_transform(embeddings0)
reduced_embeddings = reduced_embeddings0[0:embeddings.shape[0]]
reduced_test_embeddings = reduced_embeddings0[embeddings.shape[0]:]
print(reduced_embeddings.shape)
print(reduced_test_embeddings.shape)

# Assuming out_data is loaded from your JSON file
json_path = f'data/eval/{subs[i]}_num=100.json'
with open(json_path, 'r') as json_file:
    out_data = json.load(json_file)

# Define your variable names
vnames = ['gpt3', 'alpaca', 'gpt4o', 'llama1', 'llama2']
num_vars = len(vnames)
num_comparisons = 100
dim = reduced_embedding_dim = reduced_embeddings.shape[1]

# Create an empty 5x5x100 array
y = np.full((num_vars, num_vars, num_comparisons), np.nan)

# Create an empty 5x5 array for the edge information
edge = np.zeros((num_vars, num_vars))
np.fill_diagonal(edge, np.nan)
x_array = np.full((num_vars, num_vars, num_comparisons, reduced_embedding_dim), np.nan)

# Create a dictionary to map variable names to indices
name_to_index = {name: idx for idx, name in enumerate(vnames)}

# Populate the array
for entry in out_data:
    var1 = entry['variable_name1']
    var2 = entry['variable_name2']
    output = entry['GPT4_output']
    
    idx1 = name_to_index[var1]
    idx2 = name_to_index[var2]
    
    y[idx1, idx2, :] = 1 - np.array(output)
    y[idx2, idx1, :] = output  # Switch 0 and 1 for the reverse combination

    if idx1 != idx2:
        x_array[idx1, idx2, :, :] = reduced_embeddings
        x_array[idx2, idx1, :, :] = reduced_embeddings

    edge[idx1, idx2] = 1
    edge[idx2, idx1] = 1
    
# Print the resulting array shape to verify
print(y.shape)  #Y_l
print(x_array.shape) # time0
print(edge)
time0 = x_array.reshape(25, 100, dim)
Y_l = y.reshape(25,100)

L_ij = np.full((5*5,), 100)
L_val = 100
time_num = 1000
dim = reduced_embedding_dim
lambda_val = 0.00001
h = 0.3
n = 5
m = 0.1
w_iter = 100

# t_grid = generate_grid(0, 1, dim, time_num)
xi = np.random.randn(n * n, L_val, w_iter)
theta_x = np.zeros((n * n, L_val, n))
time_2d = time0.reshape(time0.shape[0] * time0.shape[1], time0.shape[2])

##### randomly choose Xim from time_2d
time_2d_filt = np.full((time_2d.shape[0],dim), np.nan)
# identify non-nan row in time_2d
non_nan_rows = ~np.isnan(time_2d).any(axis=1)
non_nan_row_indices = np.where(non_nan_rows)[0]
pp = 400
nn = non_nan_row_indices.shape[0]
rr = pp/nn
print(rr)
if rr<1:
    idx = np.random.binomial(1, rr,size = non_nan_row_indices.shape[0])
else: 
    idx = np.random.binomial(1, 1,size = non_nan_row_indices.shape[0])
# idx = np.random.binomial(1, rr,size = non_nan_row_indices.shape[0])
non_nan_row_indices_filt = non_nan_row_indices[idx==1]

idd = np.zeros(time_2d.shape[0])
idd[non_nan_row_indices_filt] = 1
idd0 = idd.reshape([n*n,L_val])

time_2d_filt[non_nan_row_indices_filt,:] = time_2d[non_nan_row_indices_filt,:]

print(time_2d_filt.shape)

####### first calculate \hat{\theta_i}(\bX_im^l)
p = 0.5
all_combinations = [(Y_l, time0, edge, L_ij, t, lambda_val, h, n, p, m) for t in time_2d_filt]

# Prepare for parallel processing
start_time_parallel = tm.time()
with ProcessPoolExecutor(max_workers=20) as executor:
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
    # ground_truth_x[ii] = res0[2]

# reshape theta_x 
theta_x = estimates_x.reshape([n*n,L_val,n])

####### then calculate \hat{\theta_i}(\xb)
all_combinations = [(Y_l, time0, edge, L_ij, t, lambda_val, h, n, p, m, w_iter, dim, xi, theta_x, idd0) for t in reduced_test_embeddings]
# Prepare for parallel processing
start_time_parallel = tm.time()

with ProcessPoolExecutor(max_workers=20) as executor:
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

time_num_new = reduced_test_embeddings.shape[0]
# Organize results into a structured format
estimates = np.zeros((time_num_new, n))
ground_truth = np.zeros((time_num_new, n))
# W is n-by-w_iter matrix among i in iter_num and t in time_num
GMB_W = np.zeros((time_num_new, n, w_iter))

for ii in range(time_num_new):
    res0 = results[ii]
#     t_all[ii] = res0[0]
    estimates[ii] = res0[1]
    # ground_truth[ii] = res0[2]
    GMB_W[ii] = res0[2]

result_dict = {'estimates': estimates, 'estimates_x': estimates_x,  'GMB_W': GMB_W}
np.savez(f'data/out/{subs[i]}_h{h}_dim{dim}_tr{embeddings.shape[0]}_te{test_embeddings.shape[0]}.npz', **result_dict)