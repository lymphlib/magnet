import magnet
import argparse
import os


def main(modeldir, meshdir):
    holes_brain_slice = magnet.io.load_mesh(meshdir + "/BrainCoronalHoles.vtu")
    color = "#F0F0F0"
    holes_brain_slice.view(figsize=(7, 7), colors=color, title="Original mesh")

    # Metis
    metis = magnet.aggmodels.METIS()
    M_metis = metis.agglomerate(holes_brain_slice, "direct_kway", nref=2**7)
    M_metis.view(figsize=(7, 7), colors=color, title="METIS")

    # Kmeans
    kmeans = magnet.aggmodels.KMEANS()
    M_kmeans = kmeans.agglomerate(holes_brain_slice, "direct_kway", nref=2**7)
    M_kmeans.view(figsize=(7, 7), colors=color, title="Kmeans")

    # SageBase2D
    sage = magnet.aggmodels.SageBase2D(64, 32, 3, 2).to(magnet.DEVICE)
    sage.load_model(modeldir + "/SAGEbase2D.pt")
    M_sage = sage.agglomerate(holes_brain_slice, "Nref", nref=7)
    M_sage.view(figsize=(7, 7), colors=color, title="SAGE-base")

    # SAGE coarse partitioner + RL refiner
    sage = magnet.aggmodels.SageBase2D(64, 32, 3, 2).to(magnet.DEVICE)
    sage.load_model(modeldir + "/SAGEbase2D_legacy.pt")  # refiner trained on this
    rlrefiner = magnet.aggmodels.Reyyy(5, 10).to(magnet.DEVICE)
    rlrefiner.load_model(modeldir + "/RLrefiner.pt")
    M_sagecp_rlr = sage.agglomerate(
        holes_brain_slice, "multilevel", nref=7, threshold=100, refiner=rlrefiner
    )
    M_sagecp_rlr.view(
        figsize=(7, 7), colors=color, title="SAGE-base coarse partitioner + RL refiner"
    )

    # RL coarse partitioner + RL coarse refiner
    rlcoarse = magnet.aggmodels.WeakContigDRLCP(64, 32, num_features=4).to(
        magnet.DEVICE
    )
    rlcoarse.load_model(modeldir + "/RLcoarsepartitioner.pt")
    rlrefiner = magnet.aggmodels.Reyyy(5, 10).to(magnet.DEVICE)
    rlrefiner.load_model(modeldir + "/RLrefiner.pt")
    M_rlcp_rlr = rlcoarse.agglomerate(
        holes_brain_slice, "multilevel", threshold=80, nref=7, refiner=rlrefiner
    )
    M_rlcp_rlr.view(
        figsize=(7, 7), colors=color, title="RL coarse partitioner + RL refiner"
    )

    # Quality metrics boxplots
    print("Computing quality metrics...")
    meshes = [M_metis, M_metis, M_sage, M_sagecp_rlr, M_rlcp_rlr]
    metrics = [m.get_quality_metrics() for m in meshes]
    legend_labels = [
        "METIS",
        "k-means",
        "SAGE-Base",
        "SAGE-Base + RL Refiner",
        "RL Partitioner + RL Refiner",
    ]
    group_labels = [
        "Circle Ratio",
        "Sphericity",
        "Uniformity Factor",
        "Volumes Difference",
    ]
    title = None
    colors = ["#5EFB6E", "#43BFC7", "#FBB917", "#8E35EF", "#C8A2C8"]
    magnet.mesh.create_grouped_boxplots(
        metrics,
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
    parser.add_argument("--meshdir", default=home + "/magnet/data")
    parser.add_argument("--modeldir", default=home + "/magnet/models")
    args = parser.parse_args()
    main(args.modeldir, args.meshdir)
