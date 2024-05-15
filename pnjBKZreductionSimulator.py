import math
from fpylll import *
from g6k import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from scipy.special import gamma

# Get the logarithms of the GSO norms of a random qary lattice 
def generate_basis_norms(dim, seed=None):
    # Set random seed for reproducibility
    if seed is None:
        seed = random.randint(0, 10000)
    FPLLL.set_random_seed(seed)

    # Generate a random qary basis
    B = IntegerMatrix.random(dim, "qary", bits = 8, k=dim)

    # Apply GSO
    M = GSO.Mat(B)
    M.update_gso()

    # Compute the norm of the basis vectors
    norms = [M.get_r(j,  j) for j in range(dim)]
    norms = np.log2(np.sqrt(norms)) # Get log of norms
    return norms

# Calculate r_i = average log |b∗_i| of a HKZ reduced random unit-volume dim-dimensional lattice
def average_log_bi(n, dim, trials):
	r = []
	for i in range(n):
		trial_norms = []
		for _ in range(trials):
			# Generate a random lattice basis
			B = IntegerMatrix.random(dim, "qary", bits = 8, k=dim)

			# Apply HKZ reduction (i.e. BKZ with beta == dim)
			M = GSO.Mat(B)
			M.update_gso()
			B_hkz = BKZ.reduction(B, BKZ.Param(block_size=dim))

			# Recalculate the GSO 
			M = GSO.Mat(B_hkz)
			M.update_gso()

			# Compute the norm of the basis vectors
			norms = [M.get_r(j, j) for j in range(dim)]
			norms = np.log2(np.sqrt(norms))
			trial_norms.append(np.mean(norms))
		r.append(np.mean(trial_norms))
	return r

# pnjBKZ simulator
def pnj_bkz_reduction_simulator(log_gram_schmidt_norms, beta, N, J, r):
    d = len(log_gram_schmidt_norms) # dimension of the lattice
    log_gram_schmidt_norms_prime = log_gram_schmidt_norms.copy()
    
    # Step 4: Compute c_i for i = 46 to d 
    c = [0] * d
    for i in range(46, d + 1):
        c[i - 1] = math.log(gamma(i / 2 + 1)**(1 / i) / math.pi**0.5)

    # Step 7: Main loop
    for j in range(N):
        flag = True # Flag to store whether L[k:d] has changed
        
        # Step 9: First nested loop
        for k in range(1, d - beta + 1):
            beta_prime = min(beta, d - k + 1) # Dimension of local block
            
            if k % J == 1:
                h = min(k + beta - 1, d) # End index of local block
                log_V = sum(log_gram_schmidt_norms[:h]) - sum(log_gram_schmidt_norms_prime[:k - 1])
                
                if flag:
                    if log_V / beta_prime + c[beta_prime - 1] < log_gram_schmidt_norms[k - 1]:
                        log_gram_schmidt_norms_prime[k - 1] = log_V / beta_prime + c[beta_prime - 1]
                        flag = False
                else:
                    log_gram_schmidt_norms_prime[k - 1] = log_V / beta_prime + c[beta_prime - 1]
            
            else:
                h = min(k - (k % J) + beta, d)
                log_V = sum(log_gram_schmidt_norms[:h]) - sum(log_gram_schmidt_norms_prime[:k - 1])
                
                if flag:
                    if log_V / (beta_prime - (k % J)) + c[beta_prime - (k % J) - 1] < log_gram_schmidt_norms[k - 1]:
                        log_gram_schmidt_norms_prime[k - 1] = log_V / (beta_prime - (k % J)) + c[beta_prime - (k % J) - 1]
                        flag = False
                else:
                    log_gram_schmidt_norms_prime[k - 1] = log_V / (beta_prime - (k % J)) + c[beta_prime - (k % J) - 1]

        # Step 35: Second nested loop
        for k in range(d - beta, d - 45):
            beta_prime = d - k # Dimension of local block
            h = d # End index of local block
            log_V = sum(log_gram_schmidt_norms[:h]) - sum(log_gram_schmidt_norms_prime[:k - 1])
            
            if flag:
                if log_V / beta_prime + c[beta_prime - 1] < log_gram_schmidt_norms[k - 1]:
                    log_gram_schmidt_norms_prime[k - 1] = log_V / beta_prime + c[beta_prime - 1]
                    flag = False
            else:
                log_gram_schmidt_norms_prime[k - 1] = log_V / beta_prime + c[beta_prime - 1]

        # Step 48: Update log_V
        log_V = sum(log_gram_schmidt_norms[:h]) - sum(log_gram_schmidt_norms_prime[:k - 1])

        # Step 49: Third nested loop
        for k in range(d - 44, d + 1):
            log_gram_schmidt_norms_prime[k - 1] = log_V / 45 + r[k + 45 - d - 1]

        # Step 52: Update log_gram_schmidt_norms for the next round
        log_gram_schmidt_norms = log_gram_schmidt_norms_prime.copy()

    # Step 56: Return the result
    return log_gram_schmidt_norms_prime


# The slope of a lattice basis based on OLS
def calculate_slope(gram_schmidt_norms):
    d = len(gram_schmidt_norms)
    b = gram_schmidt_norms
    
    x_bar = d * (d + 1) / 2
    y_bar = sum(b) / d
    
    numerator = sum(i * b[i - 1] for i in range(1, d + 1)) - d * x_bar * y_bar
    denominator = sum(i ** 2 for i in range(1, d + 1)) - d * (x_bar) ** 2
    
    slope = numerator / denominator
    return slope
     

# Example 
dim = 50 # Lattice dimesion
N = 10 #Number of tours 
jump_size = 9 #Jump value
results = []
for beta in tqdm(range(1,dim+1),desc = "Trying dimension from 1 to full dimesion...",ncols = 100):
    L = generate_basis_norms(dim)
    quality_before = calculate_slope(L)
    r = average_log_bi(45,45,10)
    L_prime = pnj_bkz_reduction_simulator(L, beta, N, jump_size,r)
    quality_after = calculate_slope(L_prime)
    results.append((quality_before,quality_after))

# Plot
# Extracting the two series from the results
x_values = [x[0] for x in results]
y_values = [x[1] for x in results]

# Scatter Plot
plt.figure(figsize=(10, 5))
width = 0.4
indices = range(len(results))
plt.bar(indices, x_values, width=width, color='b', label='Slope value before pnjBKZ')
plt.bar([i + width for i in indices], y_values, width=width, color='r', label='Slope value after pnjBKZ')
plt.title('Slopes value before/after ' +str(N) + ' pnjBKZ tours with dim = '+str(dim) +" and jump =" + str(jump_size))
plt.xlabel('Dimension')
plt.ylabel('Slope value')
plt.legend()
plt.show()
# print("Before pnjBKZ, the slope of the basis is ", quality_before,"\n After pnj BKZ with beta = ",beta," and jump = ",jump_size," the slope is ", quality_after)
