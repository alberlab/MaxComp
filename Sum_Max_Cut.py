from alabtools import analysis
from alabtools import geo
import numpy as np 
import sys
import scipy.spatial.distance as dist
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd


def plot_compartment(cp, tag):
    fig = plt.figure()
    s1 = np.where(cp >= 0)[0]
    s2 = np.where(cp < 0)[0]
    plt.bar(s1, cp[s1], color=(1.0, 0.0, 0.0), alpha=0.75, ec=None)
    plt.bar(s2, cp[s2], color=(0.0, 0.0, 1.0), alpha=0.75, ec=None)
    plt.savefig("Compartments_" + str(tag) + ".pdf", dpi=600)
# Plot compartment profile#


def radial_profile(coord, n_radius):
    rp = []

    for j in range(len(coord)):
        ratio = coord[j, :] / n_radius
        rp.append(np.linalg.norm(ratio))

    return np.array(rp)
# Compute average radial profile for every cluster#


def radius_gyration(coord, radius):
    radii = np.full(5, radius)
    gyr = []
    for i in range(len(coord) - 4):
        gyr.append(geo.RadiusOfGyration(coord[i:i + 5, :], radii))
    gyr = [0.0, 0.0] + gyr + [0.0, 0.0]

    return np.array(gyr)
# Calculate radius of gyration#


def lamina_distance(coord, n_radius):
    rt = []

    for i in range(len(coord)):
        ratio = coord[i, :]
        rt.append(n_radius - np.linalg.norm(ratio))

    return np.array(rt)
# Calculate lamina distance#


def icp(coord, length, start, end):
    matrix = dist.pdist(coord)
    matrix[matrix <= length] = 1
    matrix[matrix > length] = 0
    matrix = dist.squareform(matrix)
    matrix[start:end, start:end] = 0
    vector = np.sum(matrix, axis=1)

    return vector[start:end]
# Calculate interchromosomal contact probability#


def main():
    cell = sys.argv[1]
    chrom = int(sys.argv[2])
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    if cell == "GM":
        f = analysis.HssFile("../GM_igm-model.hss", "r")
    elif cell == "H1":
        f = analysis.HssFile("../H1_igm-model.hss", "r")
    elif cell == "HFF":
        f = analysis.HssFile("../HFF_igm-model.hss", "r")
    else:
        print("Unknown Cell Type.")
    full_coordinates = f.get_coordinates()
    radius = f.get_radii()[0]
    length = f.index.chrom_sizes
    n_radius = 5000.0
    coordinates = np.concatenate((full_coordinates[:np.sum(length[:22])], full_coordinates[np.sum(length[:24]):]), axis=0)

    cp = np.load("../" + cell + "_Compartments.npy")
    cp = cp[np.sum(length[:chrom - 1]):np.sum(length[:chrom])]
    c1 = np.where(cp >= 0)[0]
    c2 = np.where(cp < 0)[0]
    plot_compartment(cp, "GT")
    predicted_cp = np.zeros(len(cp))
    predicted_full = np.zeros(len(cp))
    predicted_full.fill(1.0)
    predicted_ratio = np.zeros(len(cp))

    spd = np.load("../" + cell + "_speckle_distance.npy")
    spd = np.concatenate((spd[:, :np.sum(length[:22])], spd[:, np.sum(length[:24]):]), axis=1)

    a_spd = []
    b_spd = []
    a_spd_gt = []
    b_spd_gt = []
    a_lmd = []
    b_lmd = []
    a_lmd_gt = []
    b_lmd_gt = []
    a_rp = []
    b_rp = []
    a_rp_gt = []
    b_rp_gt = []
    a_rg = []
    b_rg = []
    a_rg_gt = []
    b_rg_gt = []
    a_icp = []
    b_icp = []
    a_icp_gt = []
    b_icp_gt = []
    profile = []
    str_profile = []
    for i in range(start, end):
        try:
            sub_cp = np.load("compartments_" + str(i) + ".npy", allow_pickle=True)
            full_coord = coordinates[np.sum(length[:chrom - 1]):np.sum(length[:chrom]), i, :]
            full_spd = spd[i, np.sum(length[:chrom - 1]):np.sum(length[:chrom])]
            s1 = np.array(sub_cp[0]).astype(int)
            s2 = np.array(sub_cp[1]).astype(int)
            a_s = np.mean(full_spd[s1])
            b_s = np.mean(full_spd[s2])
            if a_s > b_s:
                s = np.copy(s1)
                s1 = np.copy(s2)
                s2 = np.copy(s)
            predicted_cp[s1] += 1
            predicted_ratio[s1] += 1
            sub_a_spd = np.mean(full_spd[s1])
            sub_b_spd = np.mean(full_spd[s2])
            a_spd.append(sub_a_spd)
            b_spd.append(sub_b_spd)
            sub_a_spd_gt = np.mean(full_spd[c1])
            sub_b_spd_gt = np.mean(full_spd[c2])
            a_spd_gt.append(sub_a_spd_gt)
            b_spd_gt.append(sub_b_spd_gt)
            sub_a_lmd = np.mean(lamina_distance(full_coord[s1], n_radius))
            sub_b_lmd = np.mean(lamina_distance(full_coord[s2], n_radius))
            a_lmd.append(sub_a_lmd)
            b_lmd.append(sub_b_lmd)
            sub_a_lmd_gt = np.mean(lamina_distance(full_coord[c1], n_radius))
            sub_b_lmd_gt = np.mean(lamina_distance(full_coord[c2], n_radius))
            a_lmd_gt.append(sub_a_lmd_gt)
            b_lmd_gt.append(sub_b_lmd_gt)
            sub_a_rp = np.mean(radial_profile(full_coord[s1], n_radius))
            sub_b_rp = np.mean(radial_profile(full_coord[s2], n_radius))
            a_rp.append(sub_a_rp)
            b_rp.append(sub_b_rp)
            sub_a_rp_gt = np.mean(radial_profile(full_coord[c1], n_radius))
            sub_b_rp_gt = np.mean(radial_profile(full_coord[c2], n_radius))
            a_rp_gt.append(sub_a_rp_gt)
            b_rp_gt.append(sub_b_rp_gt)
            rg = radius_gyration(full_coord, radius)
            sub_a_rg = np.mean(rg[s1][rg[s1] != 0.0])
            sub_b_rg = np.mean(rg[s2][rg[s2] != 0.0])
            a_rg.append(sub_a_rg)
            b_rg.append(sub_b_rg)
            sub_a_rg_gt = np.mean(rg[c1][rg[c1] != 0.0])
            sub_b_rg_gt = np.mean(rg[c2][rg[c2] != 0.0])
            a_rg_gt.append(sub_a_rg_gt)
            b_rg_gt.append(sub_b_rg_gt)
            sub_profile = np.zeros(len(cp))
            sub_profile[s1] = 1.0
            profile.append(sub_profile)
            str_profile.append(full_spd)
            full_genome = full_coordinates[:, i, :]
            full_icp = icp(full_genome, 1000.0, np.sum(length[:chrom - 1]), np.sum(length[:chrom]))
            sub_a_spd = np.mean(full_icp[s1])
            sub_b_spd = np.mean(full_icp[s2])
            a_icp.append(sub_a_spd)
            b_icp.append(sub_b_spd)
            sub_a_spd_gt = np.mean(full_icp[c1])
            sub_b_spd_gt = np.mean(full_icp[c2])
            a_icp_gt.append(sub_a_spd_gt)
            b_icp_gt.append(sub_b_spd_gt)
        except:
            print("File 'compartments_" + str(i) + ".npy' not found.")
            pass 

    predicted_cp /= end - start
    predicted_cp = predicted_cp - 0.5
    plot_compartment(predicted_cp, "Max_Cut")
    index = np.where(cp != 0)[0]
    cp = cp[index]
    predicted_cp = predicted_cp[index]
    r, _ = pearsonr(predicted_cp, cp)
    print(r)

    predicted_ratio /= end - start
    df_ratio = pd.DataFrame({"Beads":np.arange(len(predicted_ratio)), "Ratios":predicted_ratio})
    df_full = pd.DataFrame({"Beads":np.arange(len(predicted_full)), "Ratios":predicted_full})
    fig = plt.figure()
    sns.barplot(data=df_full, x="Ratios", y="Beads", color="darkblue", label="B", orient="h")
    sns.barplot(data=df_ratio, x="Ratios", y="Beads", color="deeppink", label="A", orient="h")
    plt.axvline(x=0.5, color="k", linestyle="--")
    plt.legend()
    plt.savefig("Ratios_Max_Cut.pdf", dpi=600)

    df = pd.DataFrame({"Max Cut":predicted_cp, "Ground Truth":cp})
    fig = plt.figure()
    sns.lmplot(data=df, x="Max Cut", y="Ground Truth", scatter_kws={"s": 20})
    plt.savefig("Correlation_Max_Cut_CP.pdf", dpi=600)

    ratios = [a_spd, b_spd]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("SpD_Box_Plot.pdf", dpi=600)
    p_value = ttest_ind(a_spd, b_spd, equal_var=False)
    print(p_value)

    ratios = [a_spd_gt, b_spd_gt]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("SpD_Box_Plot_GT.pdf", dpi=600)
    p_value = ttest_ind(a_spd_gt, b_spd_gt, equal_var=False)
    print(p_value)

    ratios = [a_lmd, b_lmd]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("LmD_Box_Plot.pdf", dpi=600)
    p_value = ttest_ind(a_lmd, b_lmd, equal_var=False)
    print(p_value)

    ratios = [a_lmd_gt, b_lmd_gt]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("LmD_Box_Plot_GT.pdf", dpi=600)
    p_value = ttest_ind(a_lmd_gt, b_lmd_gt, equal_var=False)
    print(p_value)

    ratios = [a_rp, b_rp]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("RAD_Box_Plot.pdf", dpi=600)
    p_value = ttest_ind(a_rp, b_rp, equal_var=False)
    print(p_value)

    ratios = [a_rp_gt, b_rp_gt]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("RAD_Box_Plot_GT.pdf", dpi=600)
    p_value = ttest_ind(a_rp_gt, b_rp_gt, equal_var=False)
    print(p_value)

    ratios = [a_rg, b_rg]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("RG_Box_Plot.pdf", dpi=600)
    p_value = ttest_ind(a_rg, b_rg, equal_var=False)
    print(p_value)

    ratios = [a_rg_gt, b_rg_gt]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("RG_Box_Plot_GT.pdf", dpi=600)
    p_value = ttest_ind(a_rg_gt, b_rg_gt, equal_var=False)
    print(p_value)

    ratios = [a_icp, b_icp]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("ICP_Box_Plot.pdf", dpi=600)
    p_value = ttest_ind(a_icp, b_icp, equal_var=False)
    print(p_value)

    ratios = [a_icp_gt, b_icp_gt]
    fig = plt.figure(figsize=(4, 8))
    sns.boxplot(data=ratios, showfliers=False)
    plt.savefig("ICP_Box_Plot_GT.pdf", dpi=600)
    p_value = ttest_ind(a_icp_gt, b_icp_gt, equal_var=False)
    print(p_value)

    cmap = LinearSegmentedColormap.from_list("rg", ["violet", "sienna"], N=256)
    profile = np.array(profile)
    var = np.std(profile, axis=0)
    var = var[index]
    str_var = np.log2(np.std(str_profile, axis=0) / np.mean(np.std(str_profile, axis=0)))
    str_var = str_var[index]
    fig = plt.figure()
    plt.scatter(cp, var, c=str_var, cmap=cmap)
    plt.colorbar()
    plt.savefig("Variability.pdf", dpi=600)
# Main#


if __name__ == "__main__":
    main()