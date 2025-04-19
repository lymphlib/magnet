import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gmsh
import numpy as np
import magnet


def MAGNET_logo(output_path: str):
    h=0.7
    b=0.2
    m=0.4*h
    luce=0.5
    ends=0.2
    middle=1.2
    top=1
    lc=b/3.5
    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add(output_path)
    vertices = [# inner straights
                (-luce/2, 0, 0),(luce/2, 0, 0),
                (-luce/2, ends*h, 0),(luce/2, ends*h, 0),
                (-luce/2, h-middle*b, 0),(luce/2, h-middle*b, 0),
                # outer straights
                (-luce/2-b, 0, 0),(luce/2+b, 0, 0),
                (-luce/2-b, ends*h, 0),(luce/2+b, ends*h, 0),
                (-luce/2-b, h, 0),(luce/2+b, h, 0),
                # middle top
                (-luce/2, h, 0),(luce/2, h, 0),
                (0,m,0), (0,m+middle*b,0)
                ]
    points = [gmsh.model.geo.addPoint(x, y, z, lc) for x, y, z in vertices]

    lines = [1,7,9,11,13,16,14,12,10,8,2,4,6,15,5,3]
    lines = [(lines[i], lines[i+1]) for i in range(len(lines)-1)]+[(3,1),(3,9),(10,4)]
    [gmsh.model.geo.addLine(i,j) for i, j in lines]

    bigM = (3,4,5,6,7,8,18,12,13,14,15,17)
    bigM = gmsh.model.geo.addCurveLoop(bigM)
    bigM = gmsh.model.geo.addPlaneSurface([bigM])
    gmsh.model.geo.addPhysicalGroup(2, [bigM], 0)

    squareleft = (1,2,-17,16)
    squareleft = gmsh.model.geo.addCurveLoop(squareleft)
    squareleft = gmsh.model.geo.addPlaneSurface([squareleft])
    gmsh.model.geo.addPhysicalGroup(2, [squareleft], 1)

    squareright = (10,11,-18,9)
    squareright = gmsh.model.geo.addCurveLoop(squareright)
    squareright = gmsh.model.geo.addPlaneSurface([squareright])
    gmsh.model.geo.addPhysicalGroup(2, [squareright], 2)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(output_path)
    gmsh.finalize()

if __name__ == '__main__':
    MAGNET_logo('data/logo.vtk')
    logo = magnet.io.load_mesh('data/logo.vtk')
    kmeans= magnet.aggmodels.KMEANS()

    mainM=np.arange(logo.num_cells)[logo.Physical_Groups.reshape(-1)==0]
    sq1=np.arange(logo.num_cells)[logo.Physical_Groups.reshape(-1)==1]
    sq2=np.arange(logo.num_cells)[logo.Physical_Groups.reshape(-1)==2]
    mainMsubmesh = logo.subgraph(mainM)
    agglomerated_mainM=kmeans.direct_k_way(mainMsubmesh, 12)
    classes = [mainM[i] for i in agglomerated_mainM]
    classes.extend([sq1,sq2])
    agglomerated_logo=logo._agglomeration(classes)

    colors = ['#FF1053','#66C7F4','#C1CAD6']
    # Other options
    # colors = ['tab:blue','#FF7900','#DADBDD']
    # colors = ['#4062BB','#F45B69','#EBEBEB']
    # colors = ['#65DEF1','#F96900','#DCE2C8']
    # colors = ['#ED254E','#011936','#465362']

    colors = [colors[2], colors[1], colors[0]]
    agglomerated_logo.view(view_phys=True, palette=colors, edge_color='white', line_width=5, figsize=(5,5))
