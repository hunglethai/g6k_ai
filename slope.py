from fpylll import *
from g6k import *
import matplotlib.pyplot as plt
import numpy as np

# Create a random matrix "intrel" of dimension d
d = 300
A = IntegerMatrix.random(d, "uniform", bits=8)
A_original = A
# print(A)
M = GSO.Mat(A)
M.update_gso()

# Setup the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(10, 5))  # Adjust the overall figure size as needed

# Extract and plot the squared lengths of the GSO vectors
norms = [M.get_r(i, i) for i in range(d)]
norms = np.log2(np.sqrt(norms))
axs[0].bar(range(1, d + 1), norms, color='teal')
axs[0].set_xlabel('Vector Index')
axs[0].set_ylabel('Log Length of GSO Vectors')
axs[0].set_title('Log Lengths of Consecutive GSO Vectors')
axs[0].grid(True)

# Perform LLL reduction and plot
LLL.Reduction(M)()
norms = [M.get_r(i, i) for i in range(d)]
norms = np.log2(np.sqrt(norms))
axs[1].bar(range(1, d + 1), norms, color='blue')
axs[1].set_xlabel('Vector Index')
axs[1].set_ylabel('Log Length of LLLed Vectors')
axs[1].set_title('Log Lengths of Consecutive LLLed Vectors')
axs[1].grid(True)

# # Configure and perform BKZ reduction, then plot
block_size_bkz = 20
bkz_params = BKZ.Param(block_size=block_size_bkz, strategies=BKZ.DEFAULT_STRATEGY, flags=BKZ.AUTO_ABORT | BKZ.VERBOSE)
BKZ.reduction(A_original, bkz_params)
M_bkz = GSO.Mat(A_original)
M_bkz.update_gso()
norms = [M_bkz.get_r(i, i) for i in range(d)]
norms = np.log2(np.sqrt(norms))
axs[2].bar(range(1, d + 1), norms, color='red')
axs[2].set_xlabel('Vector Index')
axs[2].set_ylabel('Log Length of BKZed Vectors')
axs[2].set_title('Log Lengths of Consecutive BKZed Vectors with block size = ' + str(block_size_bkz))
axs[2].grid(True)

# Show the plot
plt.tight_layout()
plt.show()