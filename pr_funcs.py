import numpy as np

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:

	n = A.shape[0] # taille vecteur

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


	# Système linéaire : (I - alpha*P) x = (1 - alpha) * v
	I = np.eye(n) # (matrice identité)
	x = np.linalg.solve(I - alpha * P, (1 - alpha) * v) # resolution système linéaire
	x /= x.sum()  # Normalisation final
	return x


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
	return


def randomWalk(A: np.matrix, alpha: float, v: np.array, nsteps: int = 20000, start_node: int = 0) -> np.array:
	return

