import re
import reframed
from math import inf
from reframed.solvers.solver import VarType
from reframed.solvers.solution import Solution, Status

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None


reframed_to_gpr_rxns = {}  # {reframed_rxn: [gpr_rxn1, gpr_rxn2, ...]}


def _gurobi_infinity_fix(value):
    if value == inf:
        return GRB.INFINITY
    if value == -inf:
        return -GRB.INFINITY
    return value


class GurobiMomaSolver:
    """Minimal Gurobi-only solver wrapper for quadratic MOMA."""

    status_mapping = {
        getattr(GRB, 'OPTIMAL', None): Status.OPTIMAL,
        getattr(GRB, 'SUBOPTIMAL', None): Status.SUBOPTIMAL,
        getattr(GRB, 'UNBOUNDED', None): Status.UNBOUNDED,
        getattr(GRB, 'INFEASIBLE', None): Status.INFEASIBLE,
        getattr(GRB, 'INF_OR_UNBD', None): Status.INF_OR_UNB,
    }

    def __init__(self, model):
        if gp is None or GRB is None:
            raise ImportError("gurobipy is required for alg='moma'.")
        self.model = model
        self.problem = gp.Model()
        self.problem.Params.OutputFlag = 0
        self.var_ids = []
        self.constr_ids = []
        self._build_problem(model)

    def _build_problem(self, model):
        for r_id, reaction in model.reactions.items():
            self.problem.addVar(
                name=r_id,
                lb=_gurobi_infinity_fix(reaction.lb),
                ub=_gurobi_infinity_fix(reaction.ub),
                vtype=GRB.CONTINUOUS,
            )
            self.var_ids.append(r_id)
        self.problem.update()

        table = model.metabolite_reaction_lookup()
        for m_id in model.metabolites:
            expr = gp.quicksum(
                coeff * self.problem.getVarByName(r_id)
                for r_id, coeff in table[m_id].items()
                if coeff
            )
            self.problem.addConstr(expr == 0, name=m_id)
            self.constr_ids.append(m_id)
        self.problem.update()

    def set_objective(self, linear=None, quadratic=None, minimize=True):
        lin_terms = []
        quad_terms = []

        if linear:
            for r_id, value in linear.items():
                if value:
                    lin_terms.append(value * self.problem.getVarByName(r_id))

        if quadratic:
            for (r_id1, r_id2), value in quadratic.items():
                if value:
                    quad_terms.append(value * self.problem.getVarByName(r_id1) * self.problem.getVarByName(r_id2))

        sense = GRB.MINIMIZE if minimize else GRB.MAXIMIZE
        self.problem.setObjective(gp.quicksum(lin_terms + quad_terms), sense)

    def solve(self, linear=None, quadratic=None, minimize=True, constraints=None, get_values=True):
        old_bounds = {}

        if constraints:
            for r_id, bounds in constraints.items():
                lb, ub = bounds if isinstance(bounds, tuple) else (bounds, bounds)
                var = self.problem.getVarByName(r_id)
                old_bounds[r_id] = (var.lb, var.ub)
                var.lb = _gurobi_infinity_fix(lb)
                var.ub = _gurobi_infinity_fix(ub)
            self.problem.update()

        if linear is not None or quadratic is not None:
            self.set_objective(linear, quadratic, minimize)

        self.problem.optimize()
        status = self.status_mapping.get(self.problem.status, Status.UNKNOWN)
        message = str(self.problem.status)

        if status in (Status.OPTIMAL, Status.SUBOPTIMAL):
            fobj = self.problem.ObjVal
            values = None
            if get_values:
                values = {r_id: self.problem.getVarByName(r_id).X for r_id in self.var_ids}
            solution = Solution(status, message, fobj, values)
        else:
            solution = Solution(status, message)

        if constraints:
            for r_id, (lb, ub) in old_bounds.items():
                var = self.problem.getVarByName(r_id)
                var.lb = lb
                var.ub = ub
            self.problem.update()

        return solution


def gpr_reactions(reframed_rxns, includes=[], excludes=[]):
    u_reframed_rxns = []
    r_reframed_rxns = []
    for rxn in reframed_rxns:
        if rxn.startswith('u_'):
            u_reframed_rxns.append(rxn)
        elif rxn.startswith('R_'):
            r_reframed_rxns.append(rxn)
    gpr_r_rxns = sum([reframed_to_gpr_rxns[rxn] for rxn in r_reframed_rxns], [])
    for w in includes:
        gpr_r_rxns = [rxn for rxn in gpr_r_rxns if re.match(rf"R_r_.*{w}.*$", rxn)]
    for w in excludes:
        gpr_r_rxns = [rxn for rxn in gpr_r_rxns if not re.match(rf"R_r_.*{w}.*$", rxn)]
    return sorted(gpr_r_rxns + u_reframed_rxns)


def gpr_conversion(constraints):
    gpr_constrs = {}
    for rxn, bounds in constraints.items():
        for gpr_rxn in gpr_reactions([rxn], excludes=['_f', '_b']):
            gpr_constrs.update({gpr_rxn: bounds})
        for gpr_rxn in gpr_reactions([rxn], includes=['_f']):
            gpr_constrs.update({gpr_rxn: (0, bounds[1])})
        for gpr_rxn in gpr_reactions([rxn], includes=['_b']):
            gpr_constrs.update({gpr_rxn: (0, -bounds[0])})
    return gpr_constrs


def moma_solver_instance(gpr_model, reference, alg='moma'):
    if alg == 'moma':
        solver = GurobiMomaSolver(gpr_model)
        quad_obj = {(r_id, r_id): 1 for r_id in reference.keys()}
        lin_obj = {r_id: -2 * reference[r_id] for r_id in reference.keys()}
        return solver, quad_obj, lin_obj
    solver = reframed.solvers.solver_instance(gpr_model)
    if alg == 'lmoma':
        if not hasattr(solver, 'lMOMA_flag'):
            solver.lMOMA_flag = True
            for r_id in reference.keys():
                d_pos, d_neg = r_id + '_d+', r_id + '_d-'
                solver.add_variable(d_pos, 0, float('inf'), update=False)
                solver.add_variable(d_neg, 0, float('inf'), update=False)
            solver.update()
            for r_id in reference.keys():
                d_pos, d_neg = r_id + '_d+', r_id + '_d-'
                solver.add_constraint('c' + d_pos, {r_id: -1, d_pos: 1}, '>', -reference[r_id], update=False)
                solver.add_constraint('c' + d_neg, {r_id: 1, d_neg: 1}, '>', reference[r_id], update=False)
            solver.update()
            lin_obj = dict()
            for r_id in reference.keys():
                d_pos, d_neg = r_id + '_d+', r_id + '_d-'
                lin_obj[d_pos] = 1
                lin_obj[d_neg] = 1
            return solver, lin_obj
    if alg == 'room':
        U = 1e6
        L = -1e6
        delta = 0.1
        epsilon = 0.001
        if not hasattr(solver, 'ROOM_flag'):
            solver.ROOM_flag = True

            for r_id in reference.keys():
                y_i = 'y_' + r_id
                solver.add_variable(y_i, 0, 1, vartype=VarType.BINARY, update=False)
            solver.update()

            for r_id in reference.keys():
                y_i = 'y_' + r_id
                if isinstance(reference[r_id], tuple) or isinstance(reference[r_id], list):
                    w_i_min = reference[r_id][0] if reference[r_id][0] != -float('inf') else -1000
                    w_i_max = reference[r_id][1] if reference[r_id][1] != float('inf') else 1000
                else:
                    w_i_min = reference[r_id]
                    w_i_max = reference[r_id]
                w_u = w_i_max + delta * abs(w_i_max) + epsilon
                w_l = w_i_min - delta * abs(w_i_min) - epsilon
                solver.add_constraint('c' + r_id + '_u', {r_id: 1, y_i: (w_u - U)}, '<', w_u, update=False)
                solver.add_constraint('c' + r_id + '_l', {r_id: 1, y_i: (w_l - L)}, '>', w_l, update=False)
            solver.update()

    return solver


def sum_flux(fluxes, gpr_rxns):
    return fluxes[gpr_rxns[0]].sum() + fluxes[gpr_rxns[1]].sum() - fluxes[gpr_rxns[2]].sum()
