import os
import argparse
import magnet
from magnet.mesh import create_grouped_boxplots


def main(modelsdir, datadir):
    whole_brain = magnet.io.load_mesh(datadir + "/Brain.vtu")

    metis = magnet.aggmodels.METIS()
    kmeans = magnet.aggmodels.KMEANS()
    sage = magnet.aggmodels.SageBase(128, 64, 4, 2).to(magnet.DEVICE)
    sage.load_model(modelsdir + "/SAGEbase3D.pt")
    rlrefiner = magnet.aggmodels.Reyyy(5, 10).to(magnet.DEVICE)
    rlrefiner.load_model(modelsdir + "/RLrefiner.pt")
    labels = ["METIS", "Kmeans", "SAGEbase", "SAGEbase_CP_RL_refiner"]

    print("Agglomerating meshes...")
    meshes = [
        metis.agglomerate(whole_brain, "direct_kway", nref=2**8),
        kmeans.agglomerate(whole_brain, "direct_kway", nref=2**8),
        sage.agglomerate(whole_brain, "Nref", nref=8),
        sage.agglomerate(whole_brain, "multilevel", nref=8, refiner=rlrefiner),
    ]

    # Non exploded view
    os.makedirs(datadir + "/images", exist_ok=True)
    for i, mesh in enumerate(meshes):
        magnet.io.exploded_view(
            mesh,
            scale=0,
            edge_color=None,
            edge_width=1,
            orientation=(0, 0, 0),
            save_image_path=f"{datadir}/images/WholeBrain_{labels[i]}.png",
            image_scaling=2,
            title=f"WholeBrain_{labels[i]}",
        )

    # Exploded view
    for i, mesh in enumerate(meshes):
        magnet.io.exploded_view(
            mesh,
            scale=1.5,
            edge_color=None,
            edge_width=1,
            orientation=(0, 0, 0),
            save_image_path=f"{datadir}/images/WholeBrainExploded_{labels[i]}.png",
            image_scaling=2,
            title=f"WholeBrainExploded_{labels[i]}",
        )

    # Quality metrics boxplots
    print("Computing quality metrics...")
    whole_brain_qualities = [m.get_quality_metrics() for m in meshes]
    legend_labels = [
        "METIS",
        "k-means",
        "SAGE-Base",
        "multilevel SAGE-Base + RL refiner",
    ]
    group_labels = [
        "Circle Ratio",
        "Sphericity",
        "Uniformity Factor",
        "Volumes Difference",
    ]
    title = None
    colors = ["#5EFB6E", "#43BFC7", "#FBB917", "#8E35EF", "#C8A2C8"]
    create_grouped_boxplots(
        whole_brain_qualities,
        colors=colors,
        legend_labels=legend_labels,
        group_labels=group_labels,
        title=title,
        widths=0.4,
        boxplot_spacing=0.6,
        groups_spacing=0.8,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    home = default = os.path.expanduser("~")
    parser.add_argument("--datadir", default=home + "/magnet/data")
    parser.add_argument("--modeldir", default=home + "/magnet/models")
    args = parser.parse_args()
    main(args.modeldir, args.datadir)
