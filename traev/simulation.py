from traev.utils import *
from config import *
import numpy as np
import pandas as pd
from reframed.solvers import solver_instance
from reframed.cobra.simulation import FBA, pFBA
from math import floor, ceil


def SWITCHX(model, reference=None, constraints=None, wt_constraints=None, wt_uptake_objective=None, uptake_objective=None, reactions=None, solver=None,
         fitness=1, delta=0.03, epsilon=0.001, solutions=1, use_pool=False):
    """ Run SWITCHX adapted from a Regulatory On/Off Minimization (ROOM) simulation:

    Arguments:
        model (CBModel): a constraint-based model
        reference (dict): reference flux distribution or flux ranges (optional)
        constraints (dict): environmental or additional constraints in the novel conditions (optional)
        wt_constraints (dict): constraints to calculate wild-type phenotype in the current conditions (optional)
        wt_uptake_objective (dict): rxn ids (gpr-transformed!) for components of evolution environment as keys, coefficients as values
        uptake_objective (dict): rxn ids (gpr-transformed!)for components of evolution environment as keys, coefficients as values
        reactions (list): list of reactions to include in the objective (optional, default: all)
        solver (Solver): solver instance instantiated with the model, for speed (optional)
        fitness (float): minimum fraction of maximum fitness = total uptake (default: 1)
        delta (float): relative tolerance (default: 0.03)
        epsilon (float): absolute tolerance (default: 0.001)
        solutions (int): number of solutions to compute (default: 1)
        use_pool (bool) use solver solution pool (default: False)

    Returns:
        Solution: solution

    Implementation: Adapted by Paula Jouhten from ROOM implementation of Daniel Machado, 17.2.2023
    """
    # if reference is not given an optimal wt state in current conditions is assumed
    if reference is None:
        wt_solution = pFBA(model, objective=wt_uptake_objective, constraints=wt_constraints, solver=solver, minimize=True)
        reference = wt_solution.values

    # optimal state in the novel conditions will be predicted using FBA (only optimal value of the objective function used)
    case_solution = FBA(model, objective=uptake_objective, constraints=constraints, solver=solver, minimize=True)
    uptake_optimality = (1 / fitness) * case_solution.fobj

    # uptake_objective and uptake_optimality are used for creating an optimality constraint
    U = 1e6
    if not solver:
        solver = solver_instance(model)
    if uptake_optimality < 0:
        uptake_optimality = floor(U * uptake_optimality)/U
        solver.add_constraint('c' + 'uptake_optimality', uptake_objective, '>', uptake_optimality)
    else:
        uptake_optimality = ceil(U * uptake_optimality)/U
        solver.add_constraint('c' + 'uptake_optimality', uptake_objective, '<', uptake_optimality)

    if reactions is None:
        reactions = reference.keys()

    solution = reframed.cobra.simulation.ROOM(model, 
                                            solver=solver,
                                            reference=reference,
                                            reactions=reactions,
                                            constraints=constraints,
                                            delta=delta,
                                            epsilon=epsilon,
                                            solutions=solutions,
                                            use_pool=use_pool)
    return solution


def simulate_ale_strain(gpr_model, nutrients, target_growth, extra_constraints={}, ref_fluxes=None):
    """ Simulation of ALE-evolved strain:
    
    Arguments:
        gpr_model (CBModel): GPR transformed reframed cobra model
        nutrients (list): list of nutrients in ALE environment
        target_growth (double): targeted specific growth rate (biomass)
        extra_constraints (dict): extra constraints
        ref_fluxes (dict): fluxes of reference strain
    Returns:
        solution (dict): fluxes of ALE-evolved strain
        
    """
    constraints = {rxn: (-MAX_FLUX, 0) for rxn in nutrients}
    constraints.update({r_biomass: (target_growth, MAX_FLUX)})
    constraints.update(extra_constraints)
    gpr_constraints = gpr_conversion(constraints)
    obj_rxns = gpr_reactions(nutrients, includes=['_b'])
    solution = reframed.cobra.simulation.pFBA(gpr_model,
                                              objective={rxn: 1 for rxn in obj_rxns},
                                              minimize=True,
                                              constraints=gpr_constraints)
    if ref_fluxes:
        gpr_constraints.update({rxn: (solution.to_dataframe().loc[rxn, 'value'], MAX_FLUX) for rxn in obj_rxns})
        solution = SWITCHX(gpr_model,
                        reference=ref_fluxes,
                        constraints=gpr_constraints,
                        reactions=gpr_model.u_reactions,
                        delta=1E-2,
                        epsilon=1E-4)
    
    return solution.values


def simulate_engineered_strain(gpr_model, medium, carbon_source, target_growth, max_uptake, products, extra_constraints={}, ref_fluxes=None):
    """ Simulation of engineered strain:

    Arguments:
        gpr_model (CBModel): GPR transformed reframed cobra model
        medium (list): list of nutrients in medium
        carbon_source (list): list of substrates as carbon source
        target_growth (double): targeted specific growth rate (biomass)
        max_uptake (double): max uptake of carbon source
        products (list): list of target products of engineered strain
        extra_constraints (dict): extra constraints
        ref_fluxes (dict): fluxes of reference strain
    Returns:
        solution (dict): fluxes of engineered strain

    """
    constraints = {rxn: (-MAX_FLUX, 0) for rxn in medium}
    constraints.update({rxn: (-max_uptake, 0) for rxn in carbon_source})
    constraints.update({r_biomass: (target_growth, MAX_FLUX)})
    constraints.update(extra_constraints)
    gpr_constraints = gpr_conversion(constraints)
    obj_rxns = gpr_reactions(products, includes=['_f']) + gpr_reactions(products, excludes=['_f', '_b'])
    solution = reframed.cobra.simulation.pFBA(gpr_model, 
                                                  objective={rxn: 1 for rxn in obj_rxns}, 
                                                  minimize=False, 
                                                  constraints=gpr_constraints)
    if ref_fluxes:
        gpr_constraints.update({rxn: (solution.to_dataframe().loc[rxn, 'value'], MAX_FLUX) for rxn in obj_rxns})
        solution = SWITCHX(gpr_model, 
                        reference=ref_fluxes, 
                        constraints=gpr_constraints, 
                        reactions=gpr_model.u_reactions,
                        delta=1E-2,
                        epsilon=1E-4)
        
    return solution.values


def simulate_adaptation(gpr_model, ref_fluxes, nutrients, extra_constraints={}, user_p_growth=None, user_a_growth=None):
    """ Simulation of adaptation in given environment:

    Arguments:
        gpr_model (CBModel): GPR transformed reframed cobra model
        ref_fluxes (dict): fluxes of reference strain
        nutrients (list): list of nutrients in given environment
        extra_constraints (dict): extra constraints
        user_p_growth (double): user defined specific growth rate of plastic state (optional)
        user_a_growth (double): user defined specific growth rate of adapted state (optional)
        
    Returns:
        fva_df (Dataframe): Dataframe of the concatenation of r, p and a fluxes

    """
    constraints = {rxn: (-MAX_FLUX, 0) for rxn in nutrients}
    constraints.update({r_biomass: (user_p_growth or MIN_GROWTH, ref_fluxes[r_biomass])})
    constraints.update(extra_constraints)
    gpr_constraints = gpr_conversion(constraints)
    p_solution = reframed.cobra.simulation.lMOMA(gpr_model, 
                                              reference=ref_fluxes, 
                                              constraints=gpr_constraints, 
                                              reactions=gpr_model.u_reactions)
    a_growth = user_a_growth or MAX_GROWTH
    min_obj = float('inf')
    step = 0.05
    for growth in np.arange(np.floor(a_growth / step) * step, np.ceil(p_solution.values[r_biomass] / step) * step - 1E-9, -step):
        solution = SWITCHX(gpr_model,
                        reference=ref_fluxes,
                        reactions=gpr_model.u_reactions,
                        constraints=gpr_constraints | {r_biomass:(growth, growth)},
                        uptake_objective={rxn: 1 for rxn in gpr_reactions(nutrients, includes=['_b'])},
                        delta=1E-2,
                        epsilon=1E-4)
        if solution.fobj < min_obj and solution.fobj >= 0.5:
            min_obj = solution.fobj
            a_growth = solution.values[r_biomass]
        else:
            break
    for growth in np.arange(a_growth, a_growth - 0.1 - 1E-9, -0.01):
        solution = SWITCHX(gpr_model,
                        reference=ref_fluxes,
                        reactions=gpr_model.u_reactions,
                        constraints=gpr_constraints | {r_biomass:(growth, growth)},
                        uptake_objective={rxn: 1 for rxn in gpr_reactions(nutrients, includes=['_b'])},
                        delta=1E-2,
                        epsilon=1E-4)
        if solution.fobj < min_obj and solution.fobj >= 0.5:
            min_obj = solution.fobj
            a_growth = solution.values[r_biomass]
    a_solution = SWITCHX(gpr_model,
                        reference=ref_fluxes,
                        reactions=gpr_model.u_reactions,
                        constraints=gpr_constraints | {r_biomass:(a_growth, a_growth)},
                        uptake_objective={rxn: 1 for rxn in gpr_reactions(nutrients, includes=['_b'])},
                        delta=1E-2,
                        epsilon=1E-4)
    fva_df = pd.DataFrame({'r_flux': ref_fluxes, 'p_flux': p_solution.values, 'a_flux': a_solution.values})
    return fva_df
