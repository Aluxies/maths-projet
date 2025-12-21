import numpy as np

# Calculating the probability transition matrix transposed
def probabilityTransitionMatrix(A: np.matrix):
	row_sums = A.sum(axis=1)

	row_sums[row_sums == 0] = 1 # On évite les division par 0 (dangling nodes)

	P = A / row_sums[:, np.newaxis] # normalisation
	P = P.T
	return P

# Calculating the Google matrix
def googleMatrix(P: np.matrix, alpha: float, v: np.array) -> np.matrix:
	n = P.shape[0]
	vt = v[None, :] # calculating transposition of v
	e = np.ones((n, 1)) # identity vector
	G = alpha * P.T + (1 - alpha) * (e @ vt) # google = alpha * P + (1 - alpha) e*v^T
	return G

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
	n = A.shape[0] # vector size
	P = probabilityTransitionMatrix(A)

	# Linear system : (I - alpha*P) x = (1 - alpha) * v
	I = np.eye(n) # (identity matrix)
	x = np.linalg.solve(I - alpha * P, (1 - alpha) * v) # Linear system resolution
	x /= x.sum()  # Final normalisation
	return x


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
	max_iter = 150 # max number of iterations
	var_tol = 1e-9 # variation tolerance

	P = probabilityTransitionMatrix(A) # probability transition matrix

	G = googleMatrix(P, alpha, v) # Calculates the Google matrix

	pi = A.sum(axis=0) # init : indegree per column
	pi = pi / pi.sum() # normalise the vector

	print("3 premières itérations :\n")
	for i in range(max_iter): # iterating by calculating the Google matrix
		pi_new = pi @ G
		# if variation is under the variation tolerance number we stop iterations
		if np.linalg.norm(pi_new - pi, 1) < var_tol:
			break
		# printing the 3 first iterations
		if i < 3:
			print(pi_new)
		pi = pi_new
	return pi


def randomWalk(A: np.matrix, alpha: float, v: np.array, nsteps: int = 20000, start_node: int = 0) -> np.array:
	rng = np.random.default_rng(0) # random number generator

	n = A.shape[0]
	P = probabilityTransitionMatrix(A) # probability transition matrix

	visits = np.zeros(n, dtype=float) # vector of visits count for each node

	G = googleMatrix(P, alpha, v) # Calculates the Google matrix

	current = start_node # current node index

	page_rank_linear = pageRankLinear(A, alpha, v)

	print(page_rank_linear)

	logs_every = 500
	errors = []
	k = []
	for i in range(nsteps+1):
		# teleportation based on Google matrix
		current = rng.choice(n, p=G[current])
		visits[current] += 1.0

		# Useful for generating the graph x-y
		if i != 0 and (i % logs_every == 0):
			# Calculating the mean error for this iteration
			mean_error = np.abs(page_rank_linear - pageRankRandomWalk(visits)).mean()
			errors.append(mean_error)
			k.append(i)

	return pageRankRandomWalk(visits), errors, k

# Calculating the final PageRank values based on the visits vector
def pageRankRandomWalk(visits: np.array) -> np.array:
	if visits.sum() > 0:
		distance = visits / visits.sum()
	else:
		distance = visits

	return distance
