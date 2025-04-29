import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import magnet


def create_grid_plot(
    Meshes: list[list],
    row_labels,
    col_labels,
    labels_fontsize=12,
    subplot_size=4,
    **kwargs
):
    """
    Creates an N by M grid of subplots using matplotlib.

    Parameters:
    - data: A list of lists containing N by M objects.
    - row_labels: A list of labels for the rows.
    - col_labels: A list of labels for the columns.
    - plot_function: A function that takes an axis object and an object from the data
                     and draws the appropriate subplot on that axis.

    Returns:
    - A matplotlib figure object containing the grid of subplots.
    """
    # Determine grid size
    N = len(Meshes)
    M = len(Meshes[0])

    # Create the figure and subplots
    fig, axes = plt.subplots(
        nrows=N, ncols=M, figsize=(M * subplot_size, N * subplot_size)
    )

    # Loop over the data to create each subplot
    for i in range(N):
        for j in range(M):
            Meshes[i][j].view(axes=axes[i, j], **kwargs)

    # Adding row labels on the left (vertically)
    for i, label in enumerate(row_labels):
        fig.text(
            0.045,
            (N - i * 0.87 - 0.5) / N - 0.05,
            label,
            va="center",
            ha="center",
            fontsize=labels_fontsize,
            fontweight="bold",
            rotation=90,
        )

    # Adding column labels on top
    for j, ax in enumerate(axes[0, :]):
        ax.set_title(col_labels[j], fontsize=labels_fontsize, fontweight="bold")

    # Adjust layout to make room for labels
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    # Return the figure object
    return fig


def main(modelsdir, datadir):
    # Generate the meshes with inclusions
    N_holes = [4, 16, 36, 64]  # numbers of inclusions in each mesh
    C = 0.15  # fraction of area covered by inclusions
    radii = [(C / (np.pi * N)) ** 0.5 for N in N_holes]
    lc = [R / 4 for R in radii]

    incl_names = []
    for i in range(len(N_holes)):
        incl_names.append("inclusions_" + str(N_holes[i]) + ".vtk")
        magnet.generate.circular_inclusions(
            datadir + "/" + incl_names[-1], lc[i], N_holes[i], radii[i]
        )

    # Agglomerate
    original_inclusions_meshes = [
        magnet.io.load_mesh(datadir + "/" + name) for name in incl_names
    ]
    metis = magnet.aggmodels.METIS()
    sage = magnet.aggmodels.SageBase2D(64, 32, 3, 2).to(magnet.DEVICE)
    sage.load_model(modelsdir + "/SAGEbase2D.pt")
    sagehet = magnet.aggmodels.SageHeterogeneous(64, 32, 4, 2).to(magnet.DEVICE)
    sagehet.load_model(modelsdir + "/SAGEhetero2D.pt")
    models = [metis, sage, sagehet]

    mesh_to_plot = []
    for i, mesh in enumerate(original_inclusions_meshes):
        mesh_to_plot.append([mesh])
        for j, model in enumerate(models):
            if j < 2:
                M = model.agglomerate(
                    mesh, "segregated", mult_factor=1.5 * radii[i] / 2**0.5
                )
            else:
                M = model.agglomerate(
                    mesh, "mult_factor", mult_factor=1.5 * radii[i] / 2**0.5
                )
            mesh_to_plot[i].append(M)

    # Plot
    colors = ["#FF7900", "#DADBDD"]
    col_labels = [
        "Original mesh",
        "Segregated-METIS",
        "Segregated-SAGE-Base",
        "SAGE-Heterogeneous",
    ]
    row_labels = ["N=4", "N=16", "N=36", "N=64"]
    fig = create_grid_plot(
        mesh_to_plot,
        row_labels,
        col_labels,
        subplot_size=6,
        labels_fontsize=18,
        view_phys=True,
        palette=colors,
        edge_color="black",
        line_width=0.1,
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    home = default = os.path.expanduser("~")
    parser.add_argument("--datadir", default=home + "/magnet/data")
    parser.add_argument("--modeldir", default=home + "/magnet/models")
    args = parser.parse_args()
    main(args.modeldir, args.datadir)
