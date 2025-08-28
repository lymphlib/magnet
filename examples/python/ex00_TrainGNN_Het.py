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
    
    magnet.generate.dataset_2D_hetero(
        composition={
            "circular_inclusions": 50,
            "heterogeneous_square": 50,
        },
        bounds=(50, 1500),
        output_path=outputdir,
        dataset_name=training_path.split("/")[-1],
    )
    print("Generating validation dataset..")
    magnet.generate.dataset_2D_hetero(
        composition={
            "circular_inclusions": 10,
            "heterogeneous_square": 10,
        },
        bounds=(50, 1500),
        output_path=outputdir,
        dataset_name=validation_path.split("/")[-1],
    )
    print("Loading datasets...")
    trainig_dataset = magnet.io.load_dataset(training_path)
    validation_dataset = magnet.io.load_dataset(validation_path)
    print("Sending model to device", magnet.DEVICE)
    sagehet = magnet.aggmodels.SageHeterogeneous(64, 32, 4, 2).to(magnet.DEVICE)
    print("Training")
    sagehet.train_GNN(trainig_dataset, validation_dataset, epochs=epochs, batch_size=4)

    sagehet.save_model("example_sage2D.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", default=os.path.expanduser("~") + "/Documents/magnet/data")
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()
    main(args.outputdir, args.epochs)
