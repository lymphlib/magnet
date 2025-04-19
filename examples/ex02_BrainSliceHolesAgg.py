import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import magnet

if __name__ == "__main__":
    holes_brain_slice = magnet.io.load_mesh('data/BrainCoronalHoles.vtu')
    color = '#F0F0F0'
    holes_brain_slice.view(figsize=(7,7), colors=color, title='Original mesh')

    # Metis
    metis = magnet.aggmodels.METIS()
    M_metis = metis.agglomerate(holes_brain_slice, 'Nref', nref=7)
    M_metis.view(figsize=(7,7), colors=color, title='METIS')

    # Kmeans
    kmeans = magnet.aggmodels.KMEANS()
    M_kmeans = kmeans.agglomerate(holes_brain_slice, 'Nref', nref=7)
    M_kmeans.view(figsize=(7,7), colors=color,title='Kmeans')

    # SageBase2D
    sage = magnet.aggmodels.SageBase2D(64,32,3,2).to(magnet.DEVICE)
    sage.load_model('magnet/models/SageBase2D.pt')
    M_sage = sage.agglomerate(holes_brain_slice, 'Nref', nref=7)
    M_sage.view(figsize=(7,7), colors=color,title='SAGE-base')

    # SAGE coarse partitioner + RL refiner
    sage = magnet.aggmodels.SageBase2D(64,32,3,2).to(magnet.DEVICE)
    sage.load_model('magnet/models/SAGEbase2D_legacy.pt')  # refiner trained on this 
    rlrefiner = magnet.aggmodels.Reyyy(5,10).to(magnet.DEVICE)
    rlrefiner.load_model('magnet/models/RLrefiner.pt')
    M_sagecp_rlr = sage.agglomerate(holes_brain_slice, 'multilevel', nref=7, threshold=100, refiner=rlrefiner)
    M_sagecp_rlr.view(figsize=(7,7), colors=color, title='SAGE-base coarse partitioner + RL refiner')

    # RL coarse partitioner + RL coarse refiner
    rlcoarse = magnet.aggmodels.WeakContigDRLCP(64, 32, num_features=4).to(magnet.DEVICE)
    rlcoarse.load_model('magnet/models/RLcoarsepartitioner.pt')
    rlrefiner = magnet.aggmodels.Reyyy(5,10).to(magnet.DEVICE)
    rlrefiner.load_model('magnet/models/RLrefiner.pt')
    M_rlcp_rlr = rlcoarse.agglomerate(holes_brain_slice, 'multilevel', threshold=80, nref=7, refiner=rlrefiner)
    M_rlcp_rlr.view(figsize=(7,7), colors=color, title='RL coarse partitioner + RL refiner')