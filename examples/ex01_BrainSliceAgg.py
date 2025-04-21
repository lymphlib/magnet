import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import magnet
from magnet.mesh import create_grouped_boxplots

if __name__ == "__main__":
    brain_slice = magnet.io.load_mesh('data/BrainCoronalNoHoles.vtu')
    color = '#F0F0F0'
    brain_slice.view(figsize=(7,7), colors=color, title='Original mesh')

    # Metis
    metis = magnet.aggmodels.METIS()
    M_metis = metis.agglomerate(brain_slice, 'direct_kway', nref=2**7)
    M_metis.view(figsize=(7,7), colors=color, title='METIS')

    # Kmeans
    kmeans = magnet.aggmodels.KMEANS()
    M_kmeans = kmeans.agglomerate(brain_slice, 'direct_kway', nref=2**7)
    M_kmeans.view(figsize=(7,7), colors=color, title='Kmeans')

    # SageBase2D
    sage = magnet.aggmodels.SageBase2D(64,32,3,2).to(magnet.DEVICE)
    sage.load_model('magnet/models/SAGEbase2D.pt')
    M_sage = sage.agglomerate(brain_slice, 'Nref', nref=7)
    M_sage.view(figsize=(7,7), colors=color, title='SAGE-base')

    # SAGE coarse partitioner + RL refiner
    sage = magnet.aggmodels.SageBase2D(64,32,3,2).to(magnet.DEVICE)
    sage.load_model('magnet/models/SageBase2D_legacy.pt')  # refiner trained on this 
    rlrefiner = magnet.aggmodels.Reyyy(5,10).to(magnet.DEVICE)
    rlrefiner.load_model('magnet/models/RLrefiner.pt')
    M_sagecp_rlr = sage.agglomerate(brain_slice, 'multilevel', nref=7, refiner=rlrefiner)
    M_sagecp_rlr.view(figsize=(7,7), colors=color, title='SAGE-base coarse partitioner + RL refiner')

    # RL coarse partitioner + RL coarse refiner
    rlcoarse = magnet.aggmodels.WeakContigDRLCP(64, 32, num_features=4).to(magnet.DEVICE)
    rlcoarse.load_model('magnet/models/RLcoarsepartitioner.pt')
    rlrefiner = magnet.aggmodels.Reyyy(5,10).to(magnet.DEVICE)
    rlrefiner.load_model('magnet/models/RLrefiner.pt')
    M_rlcp_rlr = rlcoarse.agglomerate(brain_slice, 'multilevel', threshold=120, nref=7, refiner=rlrefiner)
    M_rlcp_rlr.view(figsize=(7,7), colors=color, title='RL coarse partitioner + RL refiner')

    # Quality metrics box plots
    print('Computing quality metrics...')
    brain_slice_qualities = [mesh.get_quality_metrics() for mesh in [M_metis, M_kmeans, M_sage, M_sagecp_rlr, M_rlcp_rlr]]
    legend_labels = ['METIS', 'k-means', 'SAGE-Base','multilevel SAGE-Base + RL Refiner','mutlilevel RL partitioner + RL Refiner']
    group_labels = ['Circle Ratio', 'Area to Perimeter Ratio', 'Uniformity Factor', 'Volumes Difference']
    title = None
    colors = ['#5EFB6E', '#43BFC7','#FBB917','#8E35EF', '#C8A2C8']
    create_grouped_boxplots(brain_slice_qualities, colors=colors, 
                            legend_labels=legend_labels, group_labels=group_labels, 
                            title=title, label_fontsize=10,
                            widths = 0.4, boxplot_spacing=0.6, groups_spacing=0.8)