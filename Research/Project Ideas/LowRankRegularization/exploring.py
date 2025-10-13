import numpy as np
from itertools import product


def rank_norm(X):
    b = np.trace(X.T @ X)
    a = np.trace((X.T @ X).T @ (X.T @ X))
    return 1/( a / (b ** 2))

# def 3_rank_norm(X): This math doesnt work
#     b = np.trace(X.T @ X)
#     a = np.trace((X.T @ X).T @ (X))
#     return a / (b ** (3/2))


# M = np.array([
#     [ 1, 2, 3, 4],
#     [ 5, 6, 7, 8],
#     [ 9,10,11,12],
#     [13,14,15,16]
# ])

# M = np.array([
#     [  1, 0.5,   0,   0],
#     [  0,   0,   0, 0.5],
#     [  0, 0.5,   0, 0.5],
#     [  0,   0,   1,   0]
# ])

length = 10
width = 10

M = np.random.rand(length, width)
M = M/10000 #it maybe works with small numbers?

# M = M * 2

u, s, vh = np.linalg.svd(M, full_matrices=True)
S = np.zeros_like(M)
S[:len(s), :len(s)] = np.diag(s)
print("Singular values:\n", S)

# print("Increasingly low rank")
# for i in range(min(length,width)):
#     small_sigma = np.

# Rank 3 approximation
s1 = np.diag([s[0], s[1], s[2], 0])
N = u @ s1 @ vh

# Rank 2
s2 = np.diag([s[0], s[1], 0, 0])
O = u @ s2 @ vh

# Rank 1
s3 = np.diag([s[0], 0, 0, 0])
P = u @ s3 @ vh

print("Increasingly low rank")
print(rank_norm(M))
print(rank_norm(N))
print(rank_norm(O))
print(rank_norm(P))  # higher as we go down the list means good.
print("compare to schatten (nuclear) norm")
print(np.sum(s))  # lower as we go down the list means good.
print(np.sum(np.diag(s1)))
print(np.sum(np.diag(s2)))
print(np.sum(np.diag(s3)))

print("Testing random matrixes")

def low_rank_approx(matrix,rank_to_keep):
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    S_reduced = np.zeros_like(S)
    S_reduced[:rank_to_keep] = S[:rank_to_keep]
    S_reduced = np.diag(S_reduced)
    approx_matrix = np.dot(U, np.dot(S_reduced, Vh))
    return approx_matrix

def make_high_rank_matrix(M):
    
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    S[:] = S[0]
    S = np.diag(S)
    high_rank = np.dot(U, np.dot(S, Vh))
    return high_rank

print("-"*20)
for i in range(1):
    M = np.random.rand(4, 4)
    for j in range(4, 0, -1):
        M = low_rank_approx(M, j)
        print("Matrix of rank {j}:\n".format(j=j), M, "\nRank norm:", rank_norm(M))
    print("-"*20)
    print("converting to high rank")
    print("-"*20)
    M = make_high_rank_matrix(M)
    
    for j in range(4, 0, -1):
        M = low_rank_approx(M, j)
        print("Matrix of rank {j}:\n".format(j=j), M, "\nRank norm:", rank_norm(M))
