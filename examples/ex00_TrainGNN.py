import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import magnet

if __name__ == '__main__':
    # generate training and validation datasets
    magnet.generate.dataset_2D(
        composition={
            'structured_quads': 200,
            'structured_tria': 200,
            'delaunay_tria': 200,
            'voronoi_tess': 200
        },
        bounds=(50, 1500),
        output_path='data', dataset_name='training_2D_800'
    )
    magnet.generate.dataset_2D(
        composition={
            'structured_quads': 50,
            'structured_tria': 50,
            'delaunay_tria': 50,
            'voronoi_tess': 50
        },
        bounds=(50, 1500),
        output_path='data', dataset_name='validation_2D_200'
    )
    trainig_dataset = magnet.io.load_dataset('data/training_2D_800')
    validation_dataset = magnet.io.load_dataset('data/validation_2D_200')

    sage = magnet.aggmodels.SageBase2D(64,32,3,2).to(magnet.DEVICE)
    sage.train_GNN(trainig_dataset, validation_dataset, epochs=300, batch_size=4)

    sage.save_model('example_sage2D.pt')