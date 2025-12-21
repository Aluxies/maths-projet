import numpy as np
import matplotlib.pyplot as plt
from pr_funcs import pageRankLinear, pageRankPower, randomWalk, probabilityTransitionMatrix, googleMatrix


import networkx as nx
import matplotlib.pyplot as plt


def read_csv_matrix(path):
	return np.loadtxt(path, delimiter=',')

def read_personnalisation(path):
    v = np.genfromtxt(path, delimiter=',', skip_header=1)
    v = np.array(v).reshape((-1,))
    if v.sum() == 0:
        raise ValueError("Le vecteur contient que des zéros")
    return v / v.sum()


def generate_matrix_csv():
	m = np.zeros((10, 10), dtype=float)
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
	return m

def generate_plot_random_walk(errors, k):
	# Plot mean error ε(k) at time k
	plt.figure()
	plt.plot(k, errors)
	plt.xlabel("Pas de temps k")
	plt.ylabel("Erreur moyenne ε(k)")
	plt.title("Convergence de la marche aléatoire vers PageRank personnalisé")
	plt.ylim(bottom=0, top=0.02)

	plt.grid(True)
	plt.show()

if __name__ == "__main__":
	alpha = 0.9
	v = read_personnalisation("csv/VecteurPersonnalisation_Groupe19.csv")
	generate_matrix_csv()
	A = read_csv_matrix("csv/matrix_adj.csv")


	print("="*60)
	print(f"Contenu de la matrice d’adjacence: \n{A}\n")
	print(f"Contenu du vecteur : \n{v}\n\n")
	print(f"Contenu de la matrice de transition P(^T) : \n{probabilityTransitionMatrix(A)}\n\n")
	print(f"Contenu de la matrice google G : \n{googleMatrix(A, alpha, v)}\n\n")
	print("="*60)



	print("> pageRankLinear: \n")
	x1 = pageRankLinear(A, alpha, v)
	print(f"{x1}\n")
	print(f"Somme des scores (=1?) : {x1.sum()}\n",)

	# On remarque que que niveau d'importance du plus bas au plus haut est:
	# I, G, F, J, B, E, H, A, D, C

	print("> pageRankPower: \n")
	x2 = pageRankPower(A, alpha, v)
	print(f"\nx: {x2}\n")


	print("> pageRankRandomWalk: \n")
	x3, errors, k = randomWalk(A, alpha, v)
	print(f"x: {x3}\n")


	generate_plot_random_walk(errors, k)

	print("="*60)

	print("Verification avec networkx")
	G = nx.from_numpy_array(A, create_using=nx.DiGraph)
	personalization_dict = {i: v[i] for i in range(len(v))}
	pagerank_scores = nx.pagerank(G, alpha=alpha, personalization=personalization_dict)
	print(f"> {pagerank_scores}\n")

	# Partie visualisation
	min_score = x1.min()
	max_score = x1.max()
	colors = [(x1[n] - min_score) / (max_score - min_score) for n in G.nodes()]

	pos = nx.spring_layout(G, seed=42)
	fig, ax = plt.subplots(figsize=(6, 6))

	labels = {i: chr(65+i) for i in range(G.number_of_nodes())}

	nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=800, ax=ax)
	nx.draw_networkx_edges(G, pos, ax=ax)
	nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)

	sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
							norm=plt.Normalize(vmin=min_score, vmax=max_score))
	sm.set_array([])
	fig.colorbar(sm, ax=ax, label="PageRank (linear)")

	plt.axis('off')
	plt.show()
