import os
import pandas as pd
import cobra
from cobra import Reaction, Metabolite
import reframed
from config import *
from traev.utils import *


def add_reaction(rxn, r_name, lb, ub, metabolites_to_add, gene_reaction_rule=None):
    reaction = Reaction(rxn, r_name, lower_bound=lb, upper_bound=ub)
    reaction.add_metabolites(metabolites_to_add)
    if gene_reaction_rule:
        reaction.gene_reaction_rule = gene_reaction_rule
    return reaction


def add_indigoidine_pathway(cobra_model):
    ind_c = Metabolite(id='s_i001', name='indigoidine', formula='C10H12N2O4', charge=0, compartment='c')
    ind_e = Metabolite(id='s_i002', name='indigoidine', formula='C10H12N2O4', charge=0, compartment='e')
    cobra_model.add_metabolites([ind_c, ind_e])
    gln_c = cobra_model.metabolites.get_by_id('s_0999')  # L-glutamine[c]
    atp_c = cobra_model.metabolites.get_by_id('s_0434')  # ATP[c]
    o2_c = cobra_model.metabolites.get_by_id('s_1275')  # O2[c]
    # fmn_c = cobra_model.metabolites.get_by_id('s_0714')  # FMN[c]
    amp_c = cobra_model.metabolites.get_by_id('s_0423')  # AMP[c]
    ppi_c = cobra_model.metabolites.get_by_id('s_0633')  # diphosphate[c]
    # h_c = cobra_model.metabolites.get_by_id('s_0794')  # H+[c]
    h2o_c = cobra_model.metabolites.get_by_id('s_0803')  # H2O[c]
    # fmnh2_c = cobra_model.metabolites.get_by_id('s_0717')  # FMNH2[c]

    gene1 = cobra.Gene('BPSA')
    gene1.name = 'bpsA'
    cobra_model.genes.append(gene1)

    r_i001 = add_reaction('r_i001', 'indigoidine synthase', 0, 1000, {gln_c: -2.0, atp_c: -2.0, o2_c: -1.0, ind_c: 1.0, amp_c: 2.0, ppi_c: 2.0, h2o_c: 2.0}, 'BPSA')
    r_i002 = add_reaction('r_i002', 'indigoidine transport', -1000, 1000, {ind_c: -1.0, ind_e: 1.0})
    r_i003 = add_reaction('r_i003', 'indigoidine exchange', 0, 1000, {ind_e: -1.0})
    cobra_model.add_reactions([r_i001, r_i002, r_i003])
    return cobra_model

def add_bikaverin_pathway(cobra_model):
    bik_c = Metabolite(id='s_b001', name='bikaverin', formula='C20H14O8', charge=0, compartment='c')
    bik_e = Metabolite(id='s_b002', name='bikaverin', formula='C20H14O8', charge=0, compartment='e')
    cobra_model.add_metabolites([bik_c, bik_e])
    malcoa_c = cobra_model.metabolites.get_by_id('s_1101')  # malonyl-CoA[c]
    accoa_c = cobra_model.metabolites.get_by_id('s_0373')  # acetyl-CoA[c]
    sam_c = cobra_model.metabolites.get_by_id('s_1416')  # SAM[c]
    o2_c = cobra_model.metabolites.get_by_id('s_1275')  # O2[c]
    nadph_c = cobra_model.metabolites.get_by_id('s_1212')  # NADPH[c]
    # h_c = cobra_model.metabolites.get_by_id('s_0794')  # H+[c]
    coa_c = cobra_model.metabolites.get_by_id('s_0529')  # CoA[c]
    co2_c = cobra_model.metabolites.get_by_id('s_0456')  # CO2[c]
    sah_c = cobra_model.metabolites.get_by_id('s_1413')  # SAH[c]
    h2o_c = cobra_model.metabolites.get_by_id('s_0803')  # H2O[c]
    nadp_c = cobra_model.metabolites.get_by_id('s_1207')  # NADP+[c]

    gene_bik1 = cobra.Gene('BIK1')
    gene_bik1.name = 'bik1'
    gene_bik2 = cobra.Gene('BIK2')
    gene_bik2.name = 'bik2'
    gene_bik3 = cobra.Gene('BIK3')
    gene_bik3.name = 'bik3'
    cobra_model.genes += [gene_bik1, gene_bik2, gene_bik3]

    r_b001 = add_reaction('r_b001', 'bikaverin synthase', 0, 1000, {malcoa_c: -8.0, accoa_c: -1.0, sam_c: -2.0, o2_c: -2.0, nadph_c: -2.0, bik_c: 1.0, coa_c: 9.0, co2_c: 8.0, sah_c: 2.0, h2o_c: 2.0, nadp_c: 2.0}, 'BIK1 and BIK2 and BIK3')
    r_b002 = add_reaction('r_b002', 'bikaverin transport', -1000, 1000, {bik_c: -1.0, bik_e: 1.0})
    r_b003 = add_reaction('r_b003', 'bikaverin exchange', 0, 1000, {bik_e: -1.0})
    cobra_model.add_reactions([r_b001, r_b002, r_b003])
    return cobra_model

def create_gpr_model(model_file, type='aromatic'):
    reframed_model_file = f'{os.path.dirname(model_file)}/reframed-GEM-{type}.xml'

    if not os.path.exists(reframed_model_file):
        cobra_model = cobra.io.read_sbml_model(model_file)
        if type == 'aromatic':
            '''close higher alcohol acetate ester -esterases'''
            cobra_model.reactions.get_by_id('r_0656').upper_bound = 0  # isoamyl acetate-hydrolyzing esterase
            cobra_model.reactions.get_by_id('r_0657').upper_bound = 0  # isobutyl acetate-hydrolyzing esterase
            '''close phenylacetaldehyde exchange assuming redox status preferring further conversion to phenyl ethanol synthesis'''
            cobra_model.reactions.get_by_id('r_2001').upper_bound = 0  # phenylacetaldehyde exchange
        elif type == 'indigoidine':
            cobra_model = add_indigoidine_pathway(cobra_model)
        elif type == 'bikaverin':
            cobra_model = add_bikaverin_pathway(cobra_model)
        cobra.io.write_sbml_model(cobra_model, reframed_model_file)

    reframed_model = reframed.load_cbmodel(reframed_model_file, flavor='fbc2')
    reframed_model.reactions.R_r_4581.ub = 0
    reframed_model.reactions.R_r_4582.ub = 0

    '''close all environmental uptake reactions but set them reversible (able to open for a specific environment later)'''
    for rxn in substrate_rxns.values():
        reframed_model.reactions[rxn].lb = 0
        # reframed_model.reactions[rxn].ub = inf
        reframed_model.reactions[rxn].reversible = True

    gpr_model = reframed.core.transformation.gpr_transform(reframed_model, inplace=False, add_proteome=True, gene_prefix='G_', usage_prefix='u_', pseudo_genes=None)

    gpr_rxn_df = pd.DataFrame([[r[0], r[1].name] for r in gpr_model.reactions.items()], columns=['id', 'r_name']).set_index('id')
    for reframed_rxn, _ in reframed_model.reactions.items():
        reframed_to_gpr_rxns[reframed_rxn] = gpr_rxn_df[gpr_rxn_df.index.str.startswith(reframed_rxn)].index.tolist()

    return gpr_model, reframed_model
