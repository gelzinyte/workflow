import os
import time
import sys

import ase.units
import numpy as np
import spglib
from ase.constraints import ExpCellFilter
from ase.optimize.precon import PreconLBFGS

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.pressure import sample_pressure
from .utils import config_type_append
from ase.optimize import FIRE, LBFGS
from ase.utils.ff import Morse, Angle, Dihedral, VdW
from ase.calculators.ff import ForceField
from ase.optimize.precon import PreconLBFGS, Exp, FF
from ase.optimize.precon.neighbors import get_neighbours

orig_log = PreconLBFGS.log


def _new_log(self, forces=None):
    if 'buildcell_config_i' in self.atoms.info:
        if self.logfile is not None:
            self.logfile.write(str(self.atoms.info['buildcell_config_i']) + ' ')
    orig_log(self, forces)

    try:
        self.logfile.flush()
    except:
        pass


# PreconLBFGS.log = _new_log


def _run_autopara_wrappable(atoms, calculator, fmax=1.0e-3, smax=None, steps=1000, pressure=None,
           keep_symmetry=True, traj_step_interval=1, traj_subselect=None, skip_failures=True,
           results_prefix='optimize_', optimiser="PreconLBFGS", verbose=False, update_config_type=True,
           autopara_rng_seed=None, precon=None, autopara_per_item_info=None,
           **opt_kwargs):
    """runs a structure optimization 

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    fmax: float, default 1e-3
        force convergence tolerance
    smax: float, default None
        stress convergence tolerance, default from fmax
    steps: int, default 1000
        max number of steps
    pressure: None / float / tuple
        applied pressure distribution (GPa), as parsed by wfl.utils.pressure.sample_pressure()
    keep_symmetry: bool, default True
        constrain symmetry to maintain initial
    traj_step_interval: int, default 1
        if present, interval between trajectory snapshots
    traj_subselect: "last_converged", "last", "high_mace_var" default None
        rule for sub-selecting configs from the full trajectory.
        Currently implemented: "last_converged", which takes the last config, if converged.
    skip_failures: bool, default True
        just skip optimizations that raise an exception
    verbose: bool, default False
        verbose output
        optimisation logs are not printed unless this is True
    update_config_type: bool, default True
        append at.info['optimize_config_type'] at.info['config_type']
    opt_kwargs
        keyword arguments for PreconLBFGS
    autopara_rng_seed: int, default None
        global seed used to initialize rng so that each operation uses a different but
        deterministic local seed, use a random value if None

    Returns
    -------
        list(Atoms) trajectories
    """
    opt_kwargs_to_use = dict(logfile=None, master=True)
    opt_kwargs_to_use.update(opt_kwargs)

    if opt_kwargs_to_use.get('logfile') is None and verbose:
        opt_kwargs_to_use['logfile'] = '-'

    print('pre calculator')
    if isinstance(calculator, tuple) and calculator[0] == "MACE":
        from mace.calculators.mace  import MACECalculator
        calculator = (MACECalculator, calculator[1], calculator[2])

    calculator = construct_calculator_picklesafe(calculator)

    print('post calculator')

    if smax is None:
        smax = fmax

    if keep_symmetry:
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        from ase.spacegroup.symmetrize import FixSymmetry

    all_trajs = []

    for at_i, at in enumerate(atoms_to_list(atoms)):
        print("structre", at_i)
        if autopara_per_item_info is not None:
            np.random.seed(autopara_per_item_info[at_i]["rng_seed"])

        # original constraints
        org_constraints = at.constraints

        if precon == "FF":
            precon = get_ff_precon(at)
        elif precon == "Exp":
            precon = Exp(A=3) 
        elif precon==None:
            pass

        if keep_symmetry:
            sym = FixSymmetry(at)
            # Append rather than overwrite constraints
            at.set_constraint([*at.constraints, sym])

            dataset = spglib.get_symmetry_dataset((at.cell, at.get_scaled_positions(), at.numbers), 0.01)
            if 'buildcell_config_i' in at.info:
                print(at.info['buildcell_config_i'], end=' ')
            print('initial symmetry group number {}, international (Hermann-Mauguin) {} Hall {} prec {}'.format(
                dataset['number'], dataset['international'], dataset['hall'], 0.01))

        at.calc = calculator
        if pressure is not None:
            p = sample_pressure(pressure, at)
            at.info[f'optimize_pressure_GPa'] = p
            p *= ase.units.GPa
            wrapped_at = ExpCellFilter(at, scalar_pressure=p)
        else:
            wrapped_at = at

        if optimiser=="PreconLBFGS":
            opt = PreconLBFGS(wrapped_at, precon=precon, **opt_kwargs_to_use)
        elif optimiser == "LBFGS":
            opt = LBFGS(wrapped_at, **opt_kwargs_to_use)
        elif optimiser=="FIRE":
            opt = FIRE(wrapped_at, **opt_kwargs_to_use)

        # default status, will be overwritten for first and last configs in traj
        at.info['optimize_config_type'] = 'optimize_mid'
        traj = []
        cur_step = 1

        def process_step(interval):
            nonlocal cur_step
            #print(f"step {cur_step}")
            if cur_step % interval == 0:
                if 'RSS_min_vol_per_atom' in at.info and at.get_volume() / len(at) < at.info['RSS_min_vol_per_atom']:
                    raise RuntimeError('Got volume per atom {} under minimum {}'.format(at.get_volume() / len(at),
                                                                                        at.info['RSS_min_vol_per_atom']))

                if len(traj) > 0 and traj[-1] == at:
                    # Some optimization algorithms sometimes seem to repeat, perhaps
                    # only in weird circumstances, e.g. bad gradients near breakdown.
                    # Do not store those duplicate configs.
                    return
                
                if np.any(np.abs(at.get_forces()) > 100):
                    raise RuntimeError(f"Got extra large forces, interupting")

                new_config = at_copy_save_results(at, results_prefix=results_prefix)
                new_config.set_constraint(org_constraints)
                new_config.info["opt_step"] = cur_step
                traj.append(new_config)

                if "mace" in results_prefix:
                    mace_var = new_config.info[f"{results_prefix}energy_var"]
                    if mace_var > 1e-1:
                        raise RuntimeError("Too large of a variance, stopping optimisation")

            cur_step += 1

        opt.attach(process_step, 1, traj_step_interval)

        # preliminary value
        final_status = 'unconverged'

        try:
            start = time.time()
            opt.run(fmax=fmax, steps=steps)
            exec_time = time.time() - start
        except Exception as exc:
            # label actual failed optimizations
            # when this happens, the atomic config somehow ends up with a 6-vector stress, which can't be
            # read by xyz reader.
            # that should probably never happen
            final_status = 'exception'
            exec_time = "failed"
            if skip_failures:
                sys.stderr.write(f'Structure optimization failed with exception \'{exc}\'\n')
                sys.stderr.flush()
            else:
                raise

        # add time 
        if exec_time!='failed':
            time_per_step = exec_time / len(traj)
        else:
            time_per_step="failed"
        for at0 in traj:
            at0.info[f'{results_prefix}exec_time_per_step'] = time_per_step

        if len(traj) == 0 or traj[-1] != at:
            new_config = at_copy_save_results(at, results_prefix=results_prefix)
            new_config.set_constraint(org_constraints)
            traj.append(new_config)

        # set for first config, to be overwritten if it's also last config
        traj[0].info['optimize_config_type'] = 'optimize_initial'

        if opt.converged():
            final_status = 'converged'

        traj[-1].info['optimize_config_type'] = f'optimize_last_{final_status}'
        traj[-1].info['optimize_n_steps'] = opt.get_number_of_steps()

        if keep_symmetry:
            # should we check that initial is subgroup of final, i.e. no symmetry was lost?
            dataset = spglib.get_symmetry_dataset((traj[-1].cell, traj[-1].get_scaled_positions(), traj[-1].numbers), 0.01)
            if 'buildcell_config_i' in at.info:
                print(at.info['buildcell_config_i'], end=' ')
            if dataset is None:
                print('final symmetry group number None')
            else:
                print('final symmetry group number {}, international (Hermann-Mauguin) {} Hall {} prec {}'.format(
                    dataset['number'], dataset['international'], dataset['hall'], 0.01))


        if update_config_type:
            # save config_type
            for at0 in traj:
                config_type_append(at0, at0.info['optimize_config_type'])

        # Note that if resampling doesn't include original last config, later
        # steps won't be able to identify those configs as the (perhaps unconverged) minima.
        # Perhaps status should be set after resampling?
        traj = subselect_from_traj(traj, subselect=traj_subselect)

        all_trajs.append(traj)

    return all_trajs


def optimize(*args, **kwargs):
    default_autopara_info={"num_inputs_per_python_subprocess":10}

    return autoparallelize(_run_autopara_wrappable, *args, 
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(optimize, _run_autopara_wrappable, "Atoms")


# Just a placeholder for now. Could perhaps include:
#    equispaced in energy
#    equispaced in Cartesian path length
#    equispaced in some other kind of distance (e.g. SOAP)
# also, should it also have max distance instead of number of samples?
def subselect_from_traj(traj, subselect=None, results_prefix="mace_"):
    """Sub-selects configurations from trajectory.

    Parameters
    ----------
    subselect: int or string, default None

        - None: full trajectory is returned
        - int: (not implemented) how many samples to take from the trajectory.
        - str: specific method

          - "last_converged": returns [last_config] if converged, or None if not.

    """
    if subselect is None:
        return traj
    
    elif subselect=="last":
        return [traj[-1]]
    
    elif subselect=="high_mace_var":
        last_mace_var = traj[-1].info[f"{results_prefix}energy_var"]
        if last_mace_var < 1e-3:
            return [traj[-1]]
        else:
            return [at for at in traj if at.info[f"{results_prefix}energy_var"] > 1e-3]
        


    elif subselect == "last_converged":
        converged_configs = [at for at in traj if at.info["optimize_config_type"] == "optimize_last_converged"]
        if len(converged_configs) == 0:
            return None
        else:
            return converged_configs

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')



def get_ff_precon(at):


    neighbor_list = [[] for _ in range(len(at))]
    morses = []; angles = []; dihedrals = []; vdws = []

    i_list, j_list, d_list, fixed_atoms = get_neighbours(atoms=at, r_cut=1.5)
    for i, j in zip(i_list, j_list):
        neighbor_list[i].append(j)
    for i in range(len(neighbor_list)):
        neighbor_list[i].sort()

    for i in range(len(at)):
        for jj in range(len(neighbor_list[i])):
            j = neighbor_list[i][jj]
            if j > i:
                morses.append(Morse(atomi=i, atomj=j, D=6.1322, alpha=1.8502, r0=1.4322))
            for kk in range(jj+1, len(neighbor_list[i])):
                k = neighbor_list[i][kk]
                angles.append(Angle(atomi=j, atomj=i, atomk=k, k=10.0, a0=np.deg2rad(120.0), cos=True))
                for ll in range(kk+1, len(neighbor_list[i])):
                    l = neighbor_list[i][ll]
                    dihedrals.append(Dihedral(atomi=j, atomj=i, atomk=k, atoml=l, k=0.346))
    return FF(morses=morses, angles=angles, dihedrals=dihedrals)


