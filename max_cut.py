from alabtools import analysis
import numpy as np 
import sys
import scipy.spatial.distance as dist
import scipy.sparse as sp
import scipy.linalg as lg
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx 
import cvxopt as cvx
import cvxpy as cp


def adj_construction(coord, radius, cp, tag, index):
    matrix = dist.pdist(coord)
    cp_matrix = dist.pdist(cp[:, np.newaxis])
    if tag == "distance":
        matrix = distance_based(matrix, cp_matrix, radius)
    elif tag == "contact":
        matrix = contact_based(matrix, radius)
    else:
        print("Unknown transformation type.")
    plot_matrix(index, sp.csr_matrix.todense(matrix))

    return matrix.astype("float32")
# Construct a adjacent matrix#


def plot_matrix(index, matrix):
    cmap = LinearSegmentedColormap.from_list("rg", ["darkmagenta", "w", "darkcyan"], N=256)
    fig = plt.figure()
    plt.imshow(matrix, cmap=cmap, interpolation="nearest", aspect=1, vmin=0.0, vmax=0.2)
    plt.colorbar()
    plt.savefig("Matrix_" + str(index) + ".pdf", dpi=600)


def distance_based(matrix, cp_matrix, radius):
    matrix[matrix > 16 * radius] = 0.0
    exp_matrix = dist.pdist(np.arange(len(dist.squareform(matrix)))[:, np.newaxis])
    matrix = matrix / exp_matrix**(1/2)
    matrix = matrix * cp_matrix
    matrix = matrix / np.max(matrix)
    matrix = sp.csr_matrix(dist.squareform(matrix))

    return matrix
# Distance-based transformation#


def contact_based(matrix, radius):
    matrix[matrix <= 16 * radius] = 0.0
    matrix[matrix > 16 * radius] = 1.0

    return matrix
# Contact-based transformation#


def max_cut(G, N):
    L = 0.25 * nx.laplacian_matrix(G).todense()
    X = cp.Variable((N, N), PSD=True)
    cts = [cp.diag(X) == 1]

    maxcut = cp.Problem(cp.Maximize(cp.trace(L @ X)), cts)
    maxcut.solve(solver=cp.SCS)

    return X, L, maxcut
# Define the convex optimization problem#


def LDL_factorization(X):
    L, D, _ = lg.ldl(X.value)
    D[D < 0] = 0
    D = np.sqrt(D)
    L = np.matmul(L, D)

    return L
# Perform Cholesky factorization#


def random_projection(maxcut, N, V, L):
    c = 0
    obj = maxcut.value
    temp = 0

    while c < 100 or temp < 0.878 * obj:
        r = np.array(cvx.normal(N, 1))
        v = np.sign(np.matmul(V, r))
        s = np.matmul(v.T, L)
        s = np.matmul(s, v)
        if s > temp:
            v_cut = v
            temp = s
        c += 1
        
    print(temp / obj)
    v = v_cut

    return v
# Perform random projection#


def plot_compartment(index, s1, s2):
    fig = plt.figure()
    plt.bar(s1, np.ones(len(s1)) * 1, color=(1.0, 0.0, 0.0), alpha=0.75, ec=None)
    plt.bar(s2, np.ones(len(s2)) * -1, color=(0.0, 0.0, 1.0), alpha=0.75, ec=None)
    plt.savefig("Compartments_" + str(index) + ".pdf", dpi=600)
# Plot compartment profile#


def cmm_construction(index, coord, radius, s1, s2):
    o = open("Structure_" + str(index) + ".cmm", "w")
    structure = coord

    o.write('<marker_set name="marker set ' + str(0) + '">\n')

    id = 1
    for i in range(len(structure)):
        if i in s1:
            c = [1, 0, 0]
        elif i in s2:
            c = [0, 0, 1]
        o.write('<marker id="' + str(id) + '" x="' + str(structure[i][0]) + '" y="' + str(structure[i][1]) +
                '" z="' + str(structure[i][2]) + '" r="' + str(c[0]) + '" g="' + str(c[1]) + '" b="' + str(
            c[2]) + '" radius="' + str(radius) + '"/>\n')
        id += 1

    for i in range(1, id - 1):
        o.write('<link id1="' + str(i + 1) + '" id2="' + str(i) + '" r="' + str(1) + '" g="' + str(0) + '" b="' + str(
            1) + '" radius="10.0"/>\n')

    o.write('</marker_set>')
    o.close()
# Construct .cmm file for each peak structure directly from the coordinates#


def main():
    cell = sys.argv[1]
    chrom = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    if cell == "GM":
        f = analysis.HssFile("./Model/GM_igm-model.hss", "r")
    elif cell == "H1":
        f = analysis.HssFile("./Model/H1_igm-model.hss", "r")
    elif cell == "HFF":
        f = analysis.HssFile("./Model/HFF_igm-model.hss", "r")
    else:
        print("Unknown Cell Type.")
    coordinates = f.get_coordinates()
    radius = f.get_radii()[0]
    length = f.index.chrom_sizes
    n_radius = 5000.0
    coordinates = np.concatenate((coordinates[:np.sum(length[:22])], coordinates[np.sum(length[:24]):]), axis=0)

    cp = np.load("./Model/" + cell + "_compartments.npy")
    cp = cp[np.sum(length[:chrom - 1]):np.sum(length[:chrom])]
    cp1 = np.where(cp > 0)[0]
    cp2 = np.where(cp < 0)[0]
    print("Compartment A:")
    print(cp1)
    print("Compartment B:")
    print(cp2)
    plot_compartment("GT", cp1, cp2)

    spd = np.load("./Model/" + cell + "_speckle_distance.npy")
    spd = np.concatenate((spd[:, :np.sum(length[:22])], spd[:, np.sum(length[:24]):]), axis=1)
    
    for index in range(start, end):
        print("Structure: " + str(index))
        full_coord = coordinates[:, index, :]
        full_coord = full_coord[np.sum(length[:chrom - 1]):np.sum(length[:chrom]), :]
        full_spd = spd[index, np.sum(length[:chrom - 1]):np.sum(length[:chrom])]

        adj_matrix = adj_construction(full_coord, radius, full_spd, "distance", index)
        print(adj_matrix)
        G = nx.from_scipy_sparse_matrix(adj_matrix)
        N = len(G.node)

        X, L, maxcut = max_cut(G, N)

        V = LDL_factorization(X)

        x = random_projection(maxcut, N, V, L)

        s1 = [n for n in range(N) if x[n] < 0]
        s2 = [n for n in range(N) if x[n] > 0]
        inter_s1 = np.intersect1d(s1, cp1)
        inter_s2 = np.intersect1d(s2, cp1)
        if len(inter_s1) >= len(inter_s2):
            new_a = s1
            new_b = s2
        else:
            new_b = s1
            new_a = s2

        print("Predicted Compartment A:")
        print(new_a)
        print("Predicted Compartment B:")
        print(new_b)

        np.save("compartments_" + str(index) + ".npy", [new_a, new_b])
        cmm_construction(index, full_coord, radius, new_a, new_b)
        plot_compartment(index, new_a, new_b)
# Main#


if __name__ == "__main__":
    main()
