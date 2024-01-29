<div align="center">    
 
# Biorisk BOTECs   

[![PyPI - Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://github.com/PhilPalmer/biorisk-botecs)
[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/PhilPalmer/biorisk-botecs/blob/output/biorisk-botecs.ipynb)
 
</div>

This repository contains a Jupyter notebook and associated code for **back-of-the-envelope calculations (BOTECs) assessing different pandemic risks**. A rendered version of the notebook can viewed on [NBViewer](https://nbviewer.org/github/PhilPalmer/biorisk-botecs/blob/output/biorisk-botecs.ipynb).

## Setup

To run the notebook locally, you will need to install the dependencies listed in `requirements.txt`. The easiest way to do this is to use [pip](https://pip.pypa.io/en/stable/):

```bash
pip install -r requirements.txt
```

While not strictly necessary, the notebook also uses the following data files which should be present in the `data` directory:
- `Epidemics dataset 21 March 2021.xlsx` from the [Marani et al. 2021](https://www.pnas.org/doi/10.1073/pnas.2105482118) paper. This can be downloaded from [Zenodo](https://zenodo.org/records/4626111).
- `globalterrorismdb_0522dist.xlsx` from the [GTD](https://www.start.umd.edu/gtd/). This can be downloaded from [here](https://www.start.umd.edu/gtd/contact/) after sign-up.

Alternatively, you can ask me for a copy of the data files.