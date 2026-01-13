import time
import os
import sys
from traev.simulation import *
from traev.robustness_analysis import *
from modeling import *
from concurrent.futures import ProcessPoolExecutor, wait
from config import *


if os.getenv('SLURM_CPUS_PER_TASK'):
    WK_NO = int(os.getenv('SLURM_CPUS_PER_TASK'))
else:
    WK_NO = 2


ynb_medium = [substrate_rxns[sub] for sub in ['NH4']]
yp_medium = [substrate_rxns[sub] for sub in amino_acids]
mediums = {"YNB": ynb_medium, "YP": yp_medium}
a_growth = {"YNB": 0.4, "YP": 0.6}


def run_indigoidine(medium_name, nutrients, n_mutations, n_samples, output_dir):
    gpr_model, _ = create_gpr_model(model_xml, type='indigoidine')
    if not os.path.exists(f'{output_dir}/indigoidine_fva_results.csv'):
        r_fluxes = simulate_engineered_strain(gpr_model, ynb_medium, [substrate_rxns['Glc']], 0.3, 10, ['R_r_i003'])
        fva_df = simulate_adaptation(gpr_model, r_fluxes, [substrate_rxns['Gal']] + nutrients, aa_uptake_constr if medium_name == 'YP' else {}, user_a_growth=a_growth[medium_name])
        fva_df.to_csv(f'{output_dir}/indigoidine_fva_results.csv')
    else:
        fva_df = pd.read_csv(f'{output_dir}/indigoidine_fva_results.csv', index_col=0)
    ra_df = robustness_analysis(gpr_model, fva_df, [substrate_rxns['Gal']] + nutrients, ['R_r_i003'], n_mutations=n_mutations, n_samples=n_samples)
    ra_df.to_csv(f'{output_dir}/indigoidine_ra_results.csv')


def run_bikaverin(medium_name, nutrients, n_mutations, n_samples, output_dir):
    gpr_model, _ = create_gpr_model(model_xml, type='bikaverin')
    if not os.path.exists(f'{output_dir}/bikaverin_fva_results.csv'):
        r_fluxes = simulate_engineered_strain(gpr_model, ynb_medium, [substrate_rxns['Glc']], 0.3, 10, ['R_r_b003'])
        fva_df = simulate_adaptation(gpr_model, r_fluxes, [substrate_rxns['Gal']] + nutrients, aa_uptake_constr if medium_name == 'YP' else {}, user_a_growth=a_growth[medium_name])
        fva_df.to_csv(f'{output_dir}/bikaverin_fva_results.csv')
    else:
        fva_df = pd.read_csv(f'{output_dir}/bikaverin_fva_results.csv', index_col=0)
    ra_df = robustness_analysis(gpr_model, fva_df, [substrate_rxns['Gal']] + nutrients, ['R_r_b003'], n_mutations=n_mutations, n_samples=n_samples)
    ra_df.to_csv(f'{output_dir}/bikaverin_ra_results.csv')


if __name__ == '__main__':
    start_time = time.time()

    n_samples = 0
    for i, arg in enumerate(sys.argv):
        if i == 1:
            n_mutations = int(arg)
        elif i == 2:
            n_samples = int(arg)
    
    futures = []
    with ProcessPoolExecutor(max_workers=WK_NO) as executor:
        for medium_name, nutrients in mediums.items():
            output_dir = f"results/pigment/{medium_name}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            futures.append(executor.submit(run_indigoidine, medium_name, nutrients, n_mutations, n_samples, output_dir))
            futures.append(executor.submit(run_bikaverin, medium_name, nutrients, n_mutations, n_samples, output_dir))
            time.sleep(2 * 60)  # stagger job starts by 2 minutes
        wait(futures)

    print(f"Completed in {(time.time() - start_time) / 60 / 60:.2f}h.\n")
