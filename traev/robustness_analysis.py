import math
import pandas as pd
from traev.utils import *
from scipy.stats import *
from reframed.solvers.solution import Status
from config import *
import random


def mutation_sampling(single_mutations, n_mutations=5):
    mutation_samples = set()
    sample_size = min(3000, math.comb(len(single_mutations), n_mutations))
    for _ in range(sample_size * 10):
        comb = tuple(sorted(random.sample(single_mutations, min(n_mutations, len(single_mutations)))))
        mutation_samples.add(comb)
        if len(mutation_samples) >= sample_size:
            break
    return list(mutation_samples)


def robustness_analysis(gpr_model, fva_df, nutrients, desired_trait, extra_constraints={}, n_mutations=5, n_samples=100):
    """ Perform robustness analysis on the desired trait in given environment:

    Arguments:
        gpr_model (CBModel): GPR transformed reframed cobra model
        fva_df (Dataframe): Dataframe of the concatenation of r_fluxes, p_fluxes and a_fluxes
        nutrients (list): list of nutrients in the environment
        desired_trait (list): list of reactions representing the desired trait
        extra_constraints (dict): extra constraints
        n_mutations (int): number of mutations to consider
        n_samples (int): number of samples for each mutation number
    Returns:
        ra_df (Dataframe): Dataframe of robustness analysis results

    """
    single_mutations = fva_df[(fva_df.index.str.startswith('u_')) & ((fva_df['p_flux'] > 1E-4) | (fva_df['a_flux'] > 1E-4))
                              & (((fva_df['a_flux'] - fva_df['p_flux']) / fva_df['p_flux']).abs() > 1E-2)].index.tolist()

    dt_gpr_rxns = (gpr_reactions(desired_trait, excludes=['_f', '_b']), gpr_reactions(desired_trait, includes=['_f']), gpr_reactions(desired_trait, includes=['_b']))
    ut_gpr_rxns = (gpr_reactions(nutrients, excludes=['_f', '_b']), gpr_reactions(nutrients, includes=['_f']), gpr_reactions(nutrients, includes=['_b']))
    fva_df.loc['proxy_fitness', ['p_flux', 'a_flux']] = [fva_df.loc[r_biomass, 'p_flux'] / -sum_flux(fva_df['p_flux'], ut_gpr_rxns),
                                                         fva_df.loc[r_biomass, 'a_flux'] / -sum_flux(fva_df['a_flux'], ut_gpr_rxns)]
    fva_df.loc['desired_trait', ['p_flux', 'a_flux']] = [sum_flux(fva_df['p_flux'], dt_gpr_rxns) / -sum_flux(fva_df['p_flux'], ut_gpr_rxns), 
                                                         sum_flux(fva_df['a_flux'], dt_gpr_rxns) / -sum_flux(fva_df['a_flux'], ut_gpr_rxns)]
    results = []
    if n_mutations >= 1 and fva_df.loc['desired_trait', 'p_flux'] > 1E-9:
        solver, quad_obj, lin_obj = moma_solver_instance(gpr_model, fva_df.loc[gpr_model.u_reactions, 'p_flux'].to_dict())
        min_pf1 = float('inf')
        constraints = {rxn: (-MAX_FLUX, 0) for rxn in nutrients}
        constraints.update({r_biomass: (MIN_GROWTH, MAX_FLUX)})
        constraints.update(extra_constraints)
        gpr_constraints = gpr_conversion(constraints)
        for mutation in single_mutations:
            solution = solver.solve(lin_obj, 
                                quadratic=quad_obj, 
                                minimize=True, 
                                constraints=gpr_constraints | {mutation: (fva_df.loc[mutation, 'a_flux'], fva_df.loc[mutation, 'a_flux'])})
            if solution.status == Status.OPTIMAL or solution.status == Status.SUBOPTIMAL:
                pf = solution.values[r_biomass] / -sum_flux(solution.to_dataframe()['value'], ut_gpr_rxns)
                dt = sum_flux(solution.to_dataframe()['value'], dt_gpr_rxns) / -sum_flux(solution.to_dataframe()['value'], ut_gpr_rxns)
                if (pf - fva_df.loc['proxy_fitness', 'p_flux']) / fva_df.loc['proxy_fitness', 'p_flux'] > 1E-2 and pf <= fva_df.loc['proxy_fitness', 'a_flux']:
                    if 0 < dt <= fva_df.loc['desired_trait', 'p_flux']:
                        results.append([mutation, 1, pf, dt])
                        min_pf1 = min(min_pf1, pf)
        
        if len(results) > 0:
            pre_min_pf = min_pf1
            for i in range(1, n_mutations):
                comb_mutations = mutation_sampling(single_mutations, i+1)
                s_count = 0
                min_pf = float('inf')
                for mutation in comb_mutations:
                    if s_count >= n_samples:
                        break
                    solution = solver.solve(lin_obj,
                                        quadratic=quad_obj,
                                        minimize=True,
                                        constraints=gpr_constraints | {rxn: (fva_df.loc[rxn, 'a_flux'], fva_df.loc[rxn, 'a_flux']) for rxn in mutation})
                    if solution.status == Status.OPTIMAL or solution.status == Status.SUBOPTIMAL:
                        pf = solution.values[r_biomass] / -sum_flux(solution.to_dataframe()['value'], ut_gpr_rxns)
                        dt = sum_flux(solution.to_dataframe()['value'], dt_gpr_rxns) / -sum_flux(solution.to_dataframe()['value'], ut_gpr_rxns)
                        if pre_min_pf < pf <= fva_df.loc['proxy_fitness', 'a_flux'] and 0 < dt <= fva_df.loc['desired_trait', 'p_flux']:
                            results.append([mutation, len(mutation), pf, dt])
                            min_pf = min(min_pf, pf)
                            s_count += 1
                pre_min_pf = min_pf

    ra_df = pd.DataFrame(results, columns=['mutations', 'n', 'proxy_fitness', 'desired_trait']).set_index('mutations')
    ra_df.loc['p_all', ['n', 'proxy_fitness', 'desired_trait']] = [0, fva_df.loc['proxy_fitness', 'p_flux'], fva_df.loc['desired_trait', 'p_flux']]
    ra_df.loc['a_all', ['n', 'proxy_fitness', 'desired_trait']] = [len(single_mutations), fva_df.loc['proxy_fitness', 'a_flux'], fva_df.loc['desired_trait', 'a_flux']]
    return ra_df
