#!/bin/bash

export ASE_CONFIG_PATH=${HOME}/.config/ase/pytest.config.ini

# MOPAC isn't updated with the profile
export ASE_MOPAC_COMMAND="${HOME}/programs/mopac-22.1.1-linux/bin/mopac PREFIX.mop 2> /dev/null"

export JANPA_HOME_DIR="${HOME}/programs/janpa"

export AIMS_SPECIES_DIR="${HOME}/programs/fhi-aims/FHIaims_170124/species_defaults/defaults_2020/light"

OMP_NUM_THREADS=1

# Aims
pytest -v -s -rxXs  ../calculators/test_aims.py
pytest -v -s -rxXs ../calculators/test_orca.py
pytest -v -s -rxXs --basetemp ${HOME}/pytest ../test_doc_examples.py 

