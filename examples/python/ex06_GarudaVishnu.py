import argparse
import os
import magnet
from magnet.mesh import create_grouped_boxplots


def main(modeldir, datadir):
    print("Computing mesh graph...")
    GarudaVishnu = magnet.io.load_mesh(datadir + "/GarudaVishnu.vtk")

    metis = magnet.aggmodels.METIS()
    kmeans = magnet.aggmodels.KMEANS()
    sage = magnet.aggmodels.SageBase(128, 64, 4, 2).to(magnet.DEVICE)
    sage.load_model(modeldir + "/SAGEbase3D.pt")
    rlrefiner = magnet.aggmodels.Reyyy(5, 10).to(magnet.DEVICE)
    rlrefiner.load_model(modeldir + "/RLrefiner.pt")
    labels = ["METIS", "Kmeans", "SAGEbase_CP_RL_refiner"]

    print("Agglomerating with METIS...")
    # we do not use volumes as weights because METIS fails
    M1 = metis.agglomerate(GarudaVishnu, "direct_kway", nref=2**9, volume_weights=False)
    print("Agglomerating with Kmeans...")
    M2 = kmeans.agglomerate(GarudaVishnu, "direct_kway", nref=2**9)
    print("Agglomerating with SAGE-base + RL refiner...")
    try:
        M3 = sage.agglomerate(
            GarudaVishnu,
            "multilevel",
            nref=9,
            threshold=500,
            refiner=rlrefiner,
            using_cuda=True,
        )
    except:
        # The graph may be too big to fit at once into GPU memory
        M3 = sage.agglomerate(
            GarudaVishnu,
            "multilevel",
            nref=9,
            threshold=500,
            refiner=rlrefiner,
            using_cuda=False,
        )
    meshes = [M1, M2, M3]

    os.makedirs(datadir + "/images", exist_ok=True)
    for i, mesh in enumerate(meshes):
        magnet.io.exploded_view(
            mesh,
            scale=0,
            palette=None,
            edge_color=None,
            edge_width=1,
            orientation=(260, 90, 90),
            save_image_path=f"{datadir}/images/GarudaVishnu_{labels[i]}.png",
            image_scaling=2,
            title=f"GarudaVishnu_{labels[i]}",
        )

    for i, mesh in enumerate(meshes):
        magnet.io.exploded_view(
            mesh,
            scale=0.85,
            palette=None,
            edge_color=None,
            edge_width=1,
            orientation=(260, 90, 90),
            figsize=(1000, 750),
            save_image_path=f"{datadir}/images/GarudaVishnuExploded_{labels[i]}.png",
            image_scaling=2,
            title=f"GarudaVishnuExploded_{labels[i]}",
        )

    # Quality metrics box plots
    print("Computing quality metrics...")
    GV_qualities = [m.get_quality_metrics() for m in meshes]
    legend_labels = ["METIS", "k-means", "multilevel SAGE-Base + RL Refiner"]
    group_labels = [
        "Circle Ratio",
        "Sphericity",
        "Uniformity Factor",
        "Volumes Difference",
    ]
    title = None
    colors = ["#5EFB6E", "#43BFC7", "#8E35EF", "#C8A2C8"]  # ,'#FBB917'
    create_grouped_boxplots(
        GV_qualities,
        colors=colors,
        legend_labels=legend_labels,
        group_labels=group_labels,
        title=title,
        label_fontsize=10,
        widths=0.4,
        boxplot_spacing=0.6,
        groups_spacing=0.8,
        ylim=1.141,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    home = default = os.path.expanduser("~")
    parser.add_argument("--datadir", default=home + "/magnet/data")
    parser.add_argument("--modeldir", default=home + "/magnet/models")
    args = parser.parse_args()
    main(args.modeldir, args.datadir)
