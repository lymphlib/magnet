import magnet
import os
import time
import pandas as pd

def main(modeldir, n, nref, dim):
    if dim == 2:
        magnet.generate.structured_tria("tmp.vtk", (n, n))
    elif dim == 3:
        magnet.generate.generate_cube("tmp.vtk", (1 / n, 1 / n))
    else:
        raise ValueError("Unsupported dim")
    
    mesh = magnet.io.load_mesh("tmp.vtk")
    timings = {}

    def metis_kway():
        metis = magnet.aggmodels.METIS()
        metis.direct_k_way(mesh, k=2**nref)

    def metis_bisect():
        metis = magnet.aggmodels.METIS()
        metis.bisection_Nref(mesh, Nref=nref)

    def kmeans_kway():
        kmeans = magnet.aggmodels.KMEANS()
        kmeans.direct_k_way(mesh, k=2**nref)

    def kmeans_bisect():
        kmeans = magnet.aggmodels.KMEANS()
        kmeans.bisection_Nref(mesh, Nref=nref)

    def sage():
        if dim == 2:
            sage = magnet.aggmodels.SageBase2D(64, 32, 3, 2).to(magnet.DEVICE)
        elif dim == 3:
            sage = magnet.aggmodels.SageBase(128, 64, 4, 2).to(magnet.DEVICE)
        else:
            raise ValueError("Unsupported dim")
        sage.load_model(modeldir + f"/SAGEbase{dim}D.pt")
        sage.bisection_Nref(mesh, Nref=nref)

    def timeit(func, reps=10, max_dt=0.5):
        t0 = time.time()
        for _ in range(reps):
            func()
            if (time.time() - t0) > max_dt:
                reps = _ + 1
                break
        return (time.time() - t0) / reps

    timings["metis_kway"] = timeit(metis_kway)
    timings["metis_bisect"] = timeit(metis_bisect)
    timings["kmeans_kway"] = timeit(kmeans_kway)
    timings["kmeans_bisect"] = timeit(kmeans_bisect)
    timings["sage"] = timeit(sage)
    timings["num_cells"] = mesh.num_cells
    timings["dim"] = dim
    timings["nref"] = nref

    return timings

    
modeldir = os.path.expanduser("~") + "/Documents/magnet/models"

ts = []
it = 1
dim = 3
for nref in [4, 6, 8]:
    for nlvl in range(2, 13):
        n = int(2 ** (nlvl / dim))
        print("Iter: ", it)
        it += 1
        try:
            ts.append(main(modeldir, n, nref, dim))
        except Exception as e:
            print(f"{e}")

df = pd.DataFrame(ts)
df.to_csv(f"timings_{dim}D.csv", index=False)
