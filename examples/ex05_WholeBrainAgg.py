import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import magnet
from magnet.mesh import create_grouped_boxplots

if __name__ == "__main__":
    whole_brain = magnet.io.load_mesh('data/Brain.vtu')

    metis = magnet.aggmodels.METIS()
    kmeans = magnet.aggmodels.KMEANS()
    sage = magnet.aggmodels.SageBase(128,64,4,2).to(magnet.DEVICE)
    sage.load_model('magnet/models/SAGEbase3D.pt')
    rlrefiner = magnet.aggmodels.Reyyy(5,10).to(magnet.DEVICE)
    rlrefiner.load_model('magnet/models/RLrefiner.pt')
    labels=['METIS', 'Kmeans', 'SAGEbase', 'SAGEbase_CP_RL_refiner']

    print('Agglomerating meshes...')
    meshes = [
        metis.agglomerate(whole_brain, 'direct_kway', nref=256),
        kmeans.agglomerate(whole_brain, 'direct_kway', nref=256),
        sage.agglomerate(whole_brain, 'Nref', nref=8),
        sage.agglomerate(whole_brain, 'multilevel', nref=8, refiner=rlrefiner)
    ]

    # Non exploded view
    os.makedirs('data/images', exist_ok=True)
    for i, mesh in enumerate(meshes):
        magnet.io.exploded_view(mesh, scale=0,
                    edge_color=None, edge_width=1,
                    orientation=(0,0,0),
                    save_image_path=f'data/images/WholeBrain_{labels[i]}.png',
                    image_scaling=2,
                    title=f'WholeBrain_{labels[i]}')
    
    # Exploded view
    for i, mesh in enumerate(meshes): 
        magnet.io.exploded_view(mesh, scale=1.5,
                    edge_color=None, edge_width=1,
                    orientation=(0,0,0),
                    save_image_path=f'data/images/WholeBrain_exploded_{labels[i]}.png',
                    image_scaling=2,
                    title=f'WholeBrain_exploded_{labels[i]}')
    
    # QUality metrics boxplots
    print('Computing quality metrics...')
    whole_brain_qualities = [m.get_quality_metrics() for m in meshes]
    legend_labels = ['METIS', 'k-means', 'SAGE-Base','multilevel SAGE-Base + RL refiner']
    group_labels = ['Circle Ratio', 'Sphericity', 'Uniformity Factor', 'Volumes Difference']
    title = None
    colors = ['#5EFB6E', '#43BFC7','#FBB917','#8E35EF', '#C8A2C8']
    create_grouped_boxplots(whole_brain_qualities, colors=colors, legend_labels=legend_labels, group_labels=group_labels, title=title,
                            widths = 0.4, boxplot_spacing=0.6, groups_spacing=0.8)