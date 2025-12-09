import numpy as np

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:

	n = A.shape[0] # taille vecteur

	P = probabilityTransitionMatrix(A, n)

	# Système linéaire : (I - alpha*P) x = (1 - alpha) * v
	I = np.eye(n) # (matrice identité)
	x = np.linalg.solve(I - alpha * P, (1 - alpha) * v) # resolution système linéaire
	x /= x.sum()  # Normalisation final
	return x


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:

	print(A)
	# v -> personalization vector

	# max number of iterations
	max_iter = 150

	# variation tolerance
	var_tol = 1e-9

	# matrix side size
	n = A.shape[0]

	# probability transition matrix
	P = probabilityTransitionMatrix(A, n)

	print(P)

	# calculating transposition of v
	vt = v[None, :]

	# identity vector
	e = np.ones((n, 1))

	# google = alpha * P + (1 - alpha) e*v^T
	G = alpha * P.T + (1 - alpha) * (e @ vt)

	# init : indegree per column
	pi = A.sum(axis=0)
	# normalise the vector
	pi = pi / pi.sum()

	# iterating by calculating the Google matrix
	for i in range(max_iter):
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

	# random number generator
	rng = np.random.default_rng(0)

	# matrix side size
	n = A.shape[0]

	# probability transition matrix
	P = probabilityTransitionMatrix(A, n)

	# vector of visits count for each node
	visits = np.zeros(n, dtype=float)

	# calculating transposition of v
	vt = v[None, :]

	# identity vector
	e = np.ones((n, 1))

	# google = alpha * P + (1 - alpha) e*v^T
	G = alpha * P.T + (1 - alpha) * (e @ vt)

	# current node index
	current = start_node

	page_rank_linear = pageRankLinear(A, alpha, v)

	logs_every = 500

	errors = []
	k = []

	# iterating by calculating the Google matrix
	for i in range(nsteps+1):
		# teleportation based on personnalisation vector
		current = rng.choice(n, p=G[current])
		visits[current] += 1.0
		
		if i % logs_every == 0:
			# Calculating the mean error for this iteration
			mean_error = np.abs(pageRankRandomWalk(visits) - page_rank_linear).mean()
			errors.append(mean_error)
			k.append(i)
	
	return pageRankRandomWalk(visits), errors, k

def pageRankRandomWalk(visits: np.array) -> np.array:
	if visits.sum() > 0:
		distance = visits / visits.sum()
	else:
		distance = visits
	
	return distance

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
