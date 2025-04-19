import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import magnet

if __name__ == "__main__":
    Emilia = magnet.io.load_mesh('data/MeshEmilia.vtk')
    palette = [(66, 134, 244),  # Light Blue
               (244, 67, 54),   # Red
               (76, 175, 80),   # Green
               (255, 193, 7),   # Amber
               (156, 39, 176),  # Purple
               (0, 188, 212),   # Cyan
               (233, 30, 99),   # Pink
               (63, 81, 181),   # Indigo
               (139, 195, 74),  # Light Green
               (255, 152, 0),   # Orange
               (103, 58, 183),  # Deep Purple
               (3, 169, 244),   # Light Cyan
               (255, 87, 34),   # Deep Orange
               (205, 220, 57),  # Lime
               (0, 150, 136),   # Teal
               (121, 85, 72),   # Brown
               (255, 235, 59),  # Yellow
               (158, 158, 158), # Grey
               (96, 125, 139),  # Blue Grey
               (244, 143, 177), # Light Pink
               (129, 212, 250), # Light Blue
               (197, 202, 233), # Light Indigo
               (100, 221, 23),  # Light Green
               (255, 111, 0)    # Vivid Orange
               ]
    palette = [(r/255, g/255,b/255) for r,g,b in palette]
    Emilia.view(view_phys=True,line_width=0.25,palette=palette, edge_color='black',figsize=(10,10),
                title='Original mesh')

    sage = magnet.aggmodels.SageBase2D(64,32,3,2).to(magnet.DEVICE)
    sage.load_model('magnet/models/SAGEBase2D.pt')
    aggEmilia = sage.agglomerate(Emilia, 'segregated', mult_factor= 0.04)
    palette = [(255, 193, 7),   # Amber
               (76, 175, 80),   # Green
               (156, 39, 176),  # Purple
               (66, 134, 244),  # Light Blue
               (244, 67, 54),   # Red
               (0, 188, 212),   # Cyan
               (233, 30, 99),   # Pink
               ]
    palette = [(r/255, g/255,b/255) for r,g,b in palette]
    aggEmilia.view(view_phys=True,line_width=0.25, palette=palette, edge_color='black',figsize=(10,10),
                   title='Agglomerated mesh')