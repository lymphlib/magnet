import magnet
import argparse
import os


def main(outputdir, epochs):
    # generate training and validation datasets
    training_path = outputdir + "/training_2D_800"
    os.makedirs(training_path, exist_ok=True)
    validation_path = outputdir + "/validation_2D_200"
    os.makedirs(validation_path, exist_ok=True)
    print("Generating training dataset..")
    magnet.generate.dataset_2D(
        composition={
            "structured_quads": 200,
            "structured_tria": 200,
            "delaunay_tria": 200,
            "voronoi_tess": 200,
        },
        bounds=(50, 1500),
        output_path=outputdir,
        dataset_name=training_path.split("/")[-1],
    )
    print("Generating validation dataset..")
    magnet.generate.dataset_2D(
        composition={
            "structured_quads": 50,
            "structured_tria": 50,
            "delaunay_tria": 50,
            "voronoi_tess": 50,
        },
        bounds=(50, 1500),
        output_path=outputdir,
        dataset_name=validation_path.split("/")[-1],
    )
    print("Loading datasets...")
    trainig_dataset = magnet.io.load_dataset(training_path)
    validation_dataset = magnet.io.load_dataset(validation_path)
    print("Sending model to device", magnet.DEVICE)
    sage = magnet.aggmodels.SageBase2D(64, 32, 3, 2).to(magnet.DEVICE)
    print("Training")
    sage.train_GNN(trainig_dataset, validation_dataset, epochs=epochs, batch_size=4)

    sage.save_model("example_sage2D.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", default=os.path.expanduser("~") + "/magnet/data")
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()
    main(args.outputdir, args.epochs)
