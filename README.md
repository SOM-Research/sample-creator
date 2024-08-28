This repository provides the implementation and replication package for the paper titled *"On the Creation of Representative Samples of Software Repositories"* submitted to the *18th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM 2024)* conference. 

# Contents

This repository contains the source code of the tool implementation, and the data and Jupyter notebooks to replicate the running example.

## Tool Implementation

The implementation of the methodology is offered as a Python library.
It is indexed in [PyPi](https://pypi.org/project/softsampling/), published as `softsampling`.
The code of the library is in the `softsampling/` folder.
The modules have been separated by the kind of variables (i.e., numerical, categorical, or mixed), and one additional module for the preprocessing steps.

## Replication Package

The replication of the study can be performed by executing the Jupyer Notebooks in the folder `running-example`.
For each case of the running example (i.e., one numerical, one categorical, and two variables of different type) we provide a jupyter notebook to reproduce the results, along with the HTML file of the execution.
The data is in the CSV file `type_likes.csv`, containing the targeted variables of the running example: `likes` and `type`.
The dependencies can be installed using the `requirements.txt` file (i.e., the implemented python package of the paper).

We recommend creating a virtual environment (e.g. venv, conda).
The Python version must be >=3.9.

# Contributing

This project is part of a research line of the [SOM Research Lab](https://som-research.uoc.edu/) and [BESSER project](https://github.com/besser-pearl), but we are open to contributions from the community. Any comment is more than welcome!

If you are interested in contributing to this project, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Code of Conduct

At SOM Research Lab and BESSER we are dedicated to creating and maintaining welcoming, inclusive, safe, and harassment-free development spaces. Anyone participating will be subject to and agrees to sign on to our [Code of Conduct](CODE_OF_CONDUCT.md).

# Governance

The development and community management of this project follows the governance rules described in the [GOVERNANCE.md](GOVERNANCE.md) document.

# License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>

The [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license allows users to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
