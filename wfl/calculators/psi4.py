import time
import numpy as np
from ase.calculators.psi4 import Psi4 as ASE_Psi4
from .wfl_fileio_calculator import WFLFileIOCalculator
from wfl.calculators.utils import clean_rundir, save_results
from ase.calculators.calculator import all_changes, Calculator

_default_keep_files = ["*dat"]

class Psi4(WFLFileIOCalculator, ASE_Psi4):

    implemented_properties = ["energy", "forces"]
    wfl_generic_def_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(self, keep_files="default", rundir_prefix="psi4_", scratchdir=None,
                workdir=None, **kwargs):

        super().__init__(keep_files, rundir_prefix=rundir_prefix,
                    workdir=workdir, scratchdir=scratchdir, **kwargs)
                    
        self.extra_results = dict()
        # to make use of wfl.calculators.utils.save_results()
        self.extra_results["atoms"] = {}
        self.extra_results["config"] = {}

        self.orig_mult_is_None = None

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):

        self.setup_rundir()

        Calculator.calculate(self, atoms, properties, system_changes)

        if self.orig_mult_is_None is None:
            if self.parameters.get("multiplicity", None) is None:
                self.orig_mult_is_None = True  
            else:
                self.orig_mult_is_None = False

        if self.orig_mult_is_None:
            self.set(multiplicity=self.get_default_multiplicity(atoms,
                                self.parameters.get("charge", 0)))

        try:
            start = time.time()
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded=True
            if 'FAILED_PSI4' in atoms.info:
                del atoms.info['FAILED_PSI4']
            exec_time = time.time() - start
            self.extra_results["config"]["execution_time"] = exec_time
        except Exception as exc:
            atoms.info['FAILED_PSI4'] = True
            calculation_succeeded=False
            raise exc
        finally:
            # from WFLFileIOCalculator
            # self.clean_rundir(_default_keep_files, calculation_succeeded)
            pass


    @staticmethod
    def get_default_multiplicity(atoms, charge=0):
        """ Gets the multiplicity of the atoms object.

        format format: singlet = 1, doublet = 2, etc.

        Parameters
        ----------
        atoms : ase.Atoms
        charge : int

        Returns
        -------
        multiplicity : int
        """

        return int(np.sum(atoms.get_atomic_numbers() - int(charge)) % 2 + 1)
