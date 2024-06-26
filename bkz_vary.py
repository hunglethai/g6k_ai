from fpylll import *
from g6k import *
from g6k.algorithms.bkz import *
from fpylll.tools.bkz_stats import dummy_tracer
from g6k.utils.stats import SieveTreeTracer, dummy_tracer
from fpylll.tools.quality import basis_quality
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Set up a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(5, 5))  # Adjust size as needed


# Define the dimension of the lattice and the bit size for the integer matrix
dimension = 80
bits = 8

# Create a random matrix as a basis for the lattice
FPLLL.set_random_seed(1337)
A = IntegerMatrix.random(dimension, "qary", bits=bits, k = dimension//2)
A_copy = A

# GSO ORIGINAL BASIS
# Initialize the GSO object with the basis matrix
M = GSO.Mat(A)
M.update_gso()

norms = [M.get_r(i, i) for i in range(dimension)]
norms = np.log2(np.sqrt(norms))

# Plotting GSO
axs[0, 0].bar(range(1, dimension + 1), norms, color='teal')
axs[0, 0].set_xlabel('Vector Index')
axs[0, 0].set_ylabel('Log Norm of GSO Vectors')
quality = basis_quality(M)
quality_str = ', '.join(f"{key}: {value:.2f}" for key, value in quality.items())
axs[0, 0].set_title(f'Log Norms of Consecutive GSO Vectors \nBasis Quality: {quality_str}')
axs[0, 0].grid(True)

# NAIVE BKZ ALGORITHM
# Create a Siever object using the GSO matrix
g6k = Siever(A)

# Parameters for the naive BKZ tour
block_size_bkz = 20  # Size of the blocks for BKZ reduction
extra_dim4free = 0  # Extra dimensions for free reduction
trials = 1  # Number of trials or tours to run

# Running the naive BKZ tour
for _ in range(trials):
    naive_bkz_tour(g6k, dummy_tracer, blocksize=block_size_bkz, dim4free_fun= default_dim4free_fun, extra_dim4free=extra_dim4free)

tracer = SieveTreeTracer(g6k, root_label=("bkz", dimension), start_clocks=True)
stat = tracer.trace
tracer.exit()
stat = tracer.trace
print(stat)
M = GSO.Mat(A)
M.update_gso()

# Extract the squared Norms of the BKZ vectors
norms = [M.get_r(i, i) for i in range(dimension)]
norms = np.log2(np.sqrt(norms))

# Plotting BKZ naive
axs[0, 1].bar(range(1, dimension + 1), norms, color='teal')
axs[0, 1].set_xlabel('Vector Index')
axs[0, 1].set_ylabel('Log Norm of naive BKZ Vectors')
quality = basis_quality(M)
quality_str = ', '.join(f"{key}: {value:.6f}" for key, value in quality.items())
axs[0, 1].set_title('Log Norms of Consecutive naive BKZ Vectors block size of '+str(block_size_bkz)+f'\nBasis Quality: {quality_str}' )
axs[0, 1].grid(True)

# BKZ PNJ FOR DIFFERENT JUMP
# BKZ parameters for pump and jump
block_sizes = range(20,dimension,5)
extra_dim4free = 0
jumps = range(5,dimension,2) # Test jump = 5 up to the dimension
trials = 1
quality_results = []

# Run BKZ pump and jump for different jump values
for idx, block_size in tqdm(enumerate(block_sizes),ncols = 100):
    for jump in jumps:
        A = A_copy
        g6k = Siever(A)
        # print(A==A_copy)

        for _ in range(trials):
            pump_n_jump_bkz_tour(g6k, dummy_tracer,blocksize=block_size, jump = jump, dim4free_fun= default_dim4free_fun, extra_dim4free=0,
                            pump_params=None, goal_r0=0., verbose=False)

        M = GSO.Mat(A)
        M.update_gso()
        quality = basis_quality(M)

        # Store quality results
        quality_results.append((block_size, jump, quality))
    
# Print quality results as a table
df = pd.DataFrame([
    {'Block size': block_size, 'Jump': jump, **quality}
    for block_size, jump, quality in quality_results
])

# Display the DataFrame
print(df.to_string(index=False))

# EXAMPLE OF A BKZ PNJ
# Example BKZ pump and jump
dim4free_fun = int(11.5 + 0.075*block_size)
A = A_copy
g6k = Siever(A)
for _ in range(trials):
    pump_n_jump_bkz_tour(g6k, dummy_tracer,blocksize=block_sizes[0], jump = jumps[0], dim4free_fun= default_dim4free_fun, extra_dim4free=0,
                         pump_params=None, goal_r0=0., verbose=False)
M = GSO.Mat(A)
M.update_gso()

# Extract the squared Norms of the BKZ vectors
norms = [M.get_r(i, i) for i in range(dimension)]
norms = np.log2(np.sqrt(norms))

# Plotting BKZ pnj
axs[1, 1].bar(range(1, dimension + 1), norms, color='teal')
axs[1, 1].set_xlabel('Vector Index')
axs[1, 1].set_ylabel('Log Norm of BKZpnj Vectors')
quality = basis_quality(M)
quality_str = ', '.join(f"{key}: {value:.4f}" for key, value in quality.items())
axs[1, 1].set_title('Log Norms of Consecutive BKZpnj Vectors block size of '+str(block_sizes[0]) + ' and jump = ' +str(jumps[0])+f'\nBasis Quality: {quality_str}' )
axs[1, 1].grid(True)

# Leave subplot 4 empty or add another plot/info
axs[1, 0].axis('off')  # Turn off axis for unused subplot

# Display the plots
plt.tight_layout()
plt.show()
