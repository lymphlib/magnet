<p style="display: flex; align-items: center;">
  <img src="data/magnet_logo.png" alt="Logo" style="width: 50px; height: auto; margin-right: 10px;">
  <span style="font-size: 1.5em; font-weight: 700;">AGNET</span>
</p>

![Format badge](https://github.com/lymphlib/magnet/actions/workflows/format.yml/badge.svg)
![Docs badge](https://github.com/lymphlib/magnet/actions/workflows/deploy-docs.yml/badge.svg)


Magnet (**M**esh **A**glomeration by **G**raph Neural **Net**work) is an
open-source Python library that provides a simple framework for mesh agglomeration in both two- and three-dimensions. 
Magnet allows to experiment with different neural network architectures, train them, and
compare their performance to state-of-the-art methods like [METIS](https://github.com/KarypisLab/METIS) and k-means on standard quality metrics. 
Magnet can also be easily integrated with other software, in particular,
it already interfaces with [lymph](https://github.com/lymphlib/lymph).

## Get started
If you cannot wait, you can easily work with Magnet on Google colab, see [our Google Colab notebook](https://github.com/lymphlib/magnet/blob/main/examples/python/examples.ipynb) to get started!

## Documentation
Magnet is a Python package that can be installed with 

```bash
pip install .
```

However it comes with some **sharp bits**, especially in the interface with METIS and Matlab. For this reason we suggest to **read the documentation**, which is available at the [github page](http://lymphlib.github.io/magnet)

Examples on how to use Magnet are in the `examples` folder.


## Citation
If you use Magnet please cite [our pre-print](https://doi.org/10.48550/arXiv.2504.21780)

Antonietti, P.F., Caldana, M., Mazzieri, I. and Fraschini, A.R., 2025. MAGNET: an open-source library for mesh agglomeration by Graph Neural Networks. arXiv preprint arXiv:2504.21780.

## Acknowledgments
The authors acknowledge the support of the ERC Synergy Grant n. 101115663 [NEMESIS: NEw generation MEthods for numerical SImulationS](https://erc-nemesis.eu)

## Mantainers
* Matteo Caldana <matteo.caldana@polimi.it>
* Andrea Re Fraschini <andrea4.re@mail.polimi.it>