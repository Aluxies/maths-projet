import numpy as np
import matplotlib.pyplot as plt
from pr_funcs import pageRankLinear, pageRankPower, randomWalk

def read_csv_matrix(path):
	return np.loadtxt(path, delimiter=',')

def read_personnalisation(path):
    v = np.genfromtxt(path, delimiter=',', skip_header=1)
    v = np.array(v).reshape((-1,))
    if v.sum() == 0:
        raise ValueError("Le vecteur contient que des zéros")
    return v / v.sum()


def generate_matrix_csv():
	M = np.zeros((10, 10), dtype=float)
	m = [
    [0, 5, 0, 0, 0, 0, 0, 3, 0, 0], #2
    [3, 0, 1, 0, 0, 0, 0, 0, 2, 0], #3
    [0, 0, 0, 2, 0, 0, 0, 0, 2, 3], #3
    [0, 0, 3, 0, 0, 0, 0, 0, 0, 3], #2
    [0, 0, 0, 5, 0, 4, 0, 0, 0, 0], #2
    [0, 0, 0, 0, 2, 0, 5, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 2, 0, 0, 3, 0], #2
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0], #1
    [1, 4, 0, 0, 0, 0, 0, 4, 0, 4], #4
    [0, 0, 0, 0, 4, 1, 0, 0, 2, 0]] #3
	matrix = np.array(m, dtype=float)
	np.savetxt("csv/matrix_adj.csv", matrix, delimiter=",", fmt="%.6f")

	print("génération du fichier matrix csv")
	return M

def generate_plot_random_walk(errors, k):
	# Plot mean error ε(k) at time k
	plt.figure()
	plt.plot(k, errors)
	plt.xlabel("Pas de temps k")
	plt.ylabel("Erreur moyenne ε(k)")
	plt.title("Convergence de la marche aléatoire vers PageRank personnalisé")
	plt.ylim(bottom=0, top=1.1)
	
	plt.grid(True)
	plt.show()

if __name__ == "__main__":
	alpha = 0.9
	v = read_personnalisation("csv/VecteurPersonnalisation_Groupe19.csv")



	generate_matrix_csv()
	A = read_csv_matrix("csv/matrix_adj.csv")

	print("Contenu de la matrice d’adjacence: \n", A)
	print("Contenu du vecteur : \n", v)




	print("pageRankLinear: \n")
	x1 = pageRankLinear(A, alpha, v)
	print(f"x: {x1}\n")
	print("Somme des scores (=1?) :", x1.sum())

	print("pageRankPower: \n")
	x2 = pageRankPower(A, alpha, v)

	print(f"x: {x2}\n")

	print("pageRankRandomWalk: \n")
	x3, errors, k = randomWalk(A, alpha, v)

	print("")
	
	print(f"x: {x3}\n")

	generate_plot_random_walk(errors, k)

	# On remarque que que niveau d'importance du plus bas au plus haut est:
	# H, G, A, C, F, D, E, B, J, I


