from datetime import datetime
import time
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait
from traev.simulation import *
from traev.robustness_analysis import *
from modeling import *
from config import *
import dill
import ast


if os.getenv('SLURM_CPUS_PER_TASK'):
    WK_NO = int(os.getenv('SLURM_CPUS_PER_TASK'))
else:
    WK_NO = os.cpu_count() - 2


anaerobic_constr = {
    # close glycerol utilization pathway as glucose is used as c-source
    'R_r_0487': (0, 0),
    'R_r_0488': (0, 0),
    'R_r_0662': (0, 0),
    # constrain isocitrate dehydrogenase in NADPH generating direction
    'R_r_0659': (0, 1000),
    # glycine synthesis from serine when glucose is the carbon source
    'R_r_0502': (0, 1000),
    'R_r_0503': (0, 1000),
    'R_r_0732': (0, 1000),
    'R_r_0733': (0, 1000),
    # mimic anaerobiosis
    'R_r_0438': (0, 0),
    'R_r_1021': (0, 0),
    'R_r_0226': (0, 0),
    # close FMN reductases with relevance in apoptosis if these are on, anaerobic conditions generate in simulations 
    # succinate and no glycerol, this does not match with the real biological state
    'R_r_0441': (0, 0),
    'R_r_0442': (0, 0),
}
wine_must = [substrate_rxns[sub] for sub in ['Fru', 'Glc'] + nitrogen_sources]
# wine_must = [substrate_rxns[sub] for sub in ['Glc', 'NH4']]
target_growth = 0.4


def run_simulations(model_dump_file, env_id, evo_nutrients, appl_nutrients, anaerobic, output_dir):
    if not os.path.exists(f'{output_dir}/{env_id}_fva_results.csv'):
        with open(model_dump_file, 'rb') as f:
            gpr_model, rtgr = dill.load(f)
            reframed_to_gpr_rxns.update(rtgr)
        r_fluxes = simulate_ale_strain(gpr_model, evo_nutrients, target_growth)
        fva_df = simulate_adaptation(gpr_model, r_fluxes, appl_nutrients,
            (anaerobic_constr if anaerobic else {}) | {rxn: aa_uptake_constr[rxn] for rxn in appl_nutrients if rxn in aa_uptake_constr})
        fva_df.to_csv(f'{output_dir}/{env_id}_fva_results.csv')


def run_robustness_analysis(model_dump_file, env_id, nutrients, anaerobic, desired_trait, n_mutations, n_samples, output_dir):
    with open(model_dump_file, 'rb') as f:
        gpr_model, rtgr = dill.load(f)
        reframed_to_gpr_rxns.update(rtgr)
    fva_df = pd.read_csv(f'{output_dir}/{env_id}_fva_results.csv', index_col=0)
    ra_df = robustness_analysis(gpr_model, fva_df, nutrients, desired_trait,
        (anaerobic_constr if anaerobic else {}) | {rxn: aa_uptake_constr[rxn] for rxn in nutrients if rxn in aa_uptake_constr}, n_mutations, n_samples)
    ra_df.to_csv(f'{output_dir}/{env_id}_ra_results.csv')


if __name__ == '__main__':
    start_time = time.time()

    env_df = pd.read_csv('data/ale_envs.csv', index_col=0)
    dt_df = pd.read_csv('data/dt_aromatic.csv', index_col=0)
    n_samples = 0
    output_dir = "results/aromatic"
    for i, arg in enumerate(sys.argv):
        if i == 1:
            target_aroma = arg
        elif i == 2:
            anaerobic = True if arg == 'Y' else False
        elif i == 3:
            try:
                with open(arg, 'r') as env_file:
                    env_df = pd.read_csv(env_file, index_col=0)
            except IOError:
                env_df = env_df.loc[arg.split(','), :]
        elif i == 4:
            n_mutations = int(arg)
        elif i == 5:
            n_samples = int(arg)
        elif i == 6:
            output_dir = arg

    desired_trait = dt_df[dt_df[f'DT_{target_aroma}'] != 0].index.tolist()
    evo_envs = [(env_id, ast.literal_eval(row['exchange_reactions'])) for env_id, row in env_df.iterrows()]
    
    output_dir = f"{output_dir}/{target_aroma}_{'ana' if anaerobic else 'aer'}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gpr_model, _ = create_gpr_model(model_xml, type='aromatic')
    model_dump_file = f"data/aromatic_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl"
    with open(model_dump_file, 'wb') as f:
        dill.dump((gpr_model, reframed_to_gpr_rxns), f)

    with ProcessPoolExecutor(max_workers=WK_NO) as executor:
        futures = [executor.submit(run_simulations, model_dump_file, env_id, evo_nutrients, wine_must, anaerobic, output_dir) for env_id, evo_nutrients in evo_envs]
        wait(futures)
    
    with ProcessPoolExecutor(max_workers=WK_NO) as executor:
        futures = [executor.submit(run_robustness_analysis, model_dump_file, env_id, wine_must, anaerobic, desired_trait, n_mutations, n_samples, output_dir) for env_id, _ in evo_envs]
        wait(futures)
            
    if os.path.exists(model_dump_file):
        os.remove(model_dump_file)
    print(f"Completed in {(time.time() - start_time) / 60 / 60:.2f}h.\n")
    