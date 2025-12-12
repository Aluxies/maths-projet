import numpy as np

def probabilityTransitionMatrix(A: np.matrix, n: int):
	col_sums = A.sum(axis=0) # somme des columns
	#print("col_sums",col_sums)

	# Matrice de transition P : on normalise chaque colonne de A pour qu'elle fasse 1
	# On évite les division par 0 (dangling nodes)
	P = np.zeros_like(A, dtype=float)
	for j in range(n):
		if col_sums[j] > 0:
			P[:, j] = A[:, j] / col_sums[j] # normalisation column (somme=1)
		else:
			P[:, j] = 1.0 / n  # pour celles vides, on les redistribue uniformément

	#print("p",P)
	return P

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
	n = A.shape[0] # taille vecteur
	P = probabilityTransitionMatrix(A, n)

	# Système linéaire : (I - alpha*P) x = (1 - alpha) * v
	I = np.eye(n) # (matrice identité)
	x = np.linalg.solve(I - alpha * P, (1 - alpha) * v) # resolution système linéaire
	x /= x.sum()  # Normalisation final
	return x


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
	max_iter = 150 # max number of iterations
	var_tol = 1e-9 # variation tolerance

	n = A.shape[0] # matrix side size
	P = probabilityTransitionMatrix(A, n) # probability transition matrix

	vt = v[None, :] # calculating transposition of v
	e = np.ones((n, 1)) # identity vector
	G = alpha * P.T + (1 - alpha) * (e @ vt) # google = alpha * P + (1 - alpha) e*v^T

	pi = A.sum(axis=0) # init : indegree per column
	pi = pi / pi.sum() # normalise the vector

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

	n = A.shape[0] # matrix side size
	P = probabilityTransitionMatrix(A, n) # probability transition matrix

	visits = np.zeros(n, dtype=float) # vector of visits count for each node

	vt = v[None, :] # calculating transposition of v
	e = np.ones((n, 1)) # identity vector
	G = alpha * P.T + (1 - alpha) * (e @ vt) # google = alpha * P + (1 - alpha) e*v^T

	current = start_node # current node index

	page_rank_linear = pageRankLinear(A, alpha, v)

	print(page_rank_linear)

	logs_every = 500
	errors = []
	k = []
	for i in range(nsteps+1): # iterating by calculating the Google matrix
		# teleportation based on personnalisation vector
		current = rng.choice(n, p=G[current])
		visits[current] += 1.0

		if i != 0 and (i % logs_every == 0):
			# Calculating the mean error for this iteration
			mean_error = 1 / (1 + np.abs(page_rank_linear - pageRankRandomWalk(visits)).mean())
			errors.append(mean_error)
			k.append(i)

	return pageRankRandomWalk(visits), errors, k


def pageRankRandomWalk(visits: np.array) -> np.array:
	if visits.sum() > 0:
		print(visits.sum())
		distance = visits / visits.sum()
	else:
		distance = visits

	return distance
