# Magnet

![Magnet logo](/data/magnet_logo.png)

Magnet (**M**esh **A**glomeration by **G**raph Neural **Net**work) is an
open-source Python library that provides a simple framework for mesh agglomeration in both two- and three-dimensions. 
Magnet allows to experiment with different neural network architectures, train them, and
compare their performance to state-of-the-art methods like [METIS](https://github.com/KarypisLab/METIS) and k-means on standard quality metrics. 
Magnet can also be easily integrated with other software, in particular,
it already interfaces with [lymph](https://github.com/lymphlib/lymph).

## Documentation
If you cannot wait, you can easily work with Magnet on Google colab, see [our Google Colab notebook](https://github.com/lymphlib/magnet/blob/main/examples/python/examples.ipynb) to get started!

Magnet comes as a Python package that can be installed with 

```bash
pip install .
```

However it comes with some sharp bits, especially in the interface with METIS and Matlab. For this reaso we suggest to read the **full documentation**, which is available at the [github page](http://lymphlib.github.io/magnet)

Examples on how to use Magnet are in the `examples` folder.


## Citation
If you use Magnet please cite [our pre-print](TODO)

### Acknowledgments
The authors acknowledge the support of the ERC Synergy Grant n. 101115663 [NEMESIS: NEw generation MEthods for numerical SImulationS](https://erc-nemesis.eu)

### Mantainers
* Matteo Caldana <matteo.caldana@polimi.it>
* Andrea Re Fraschini <andrea4.re@mail.polimi.it>