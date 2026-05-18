import os
import re
from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from traev.constants import carbon_sources, substrate_symbols


substrate_symbols = substrate_symbols | {"ammonium": r"NH$_4$"}


def to_env_name(env_id, env_df=None):
    if env_df is not None and env_id in env_df.index:
        nutrient_names = [substrate_symbols[nut] for nut in env_df.loc[env_id, "nutrients"].split(", ")]
        nutrient_names = sorted(nutrient_names, key=lambda x: (0, x) if x in carbon_sources else (1, x))
        return f"{','.join(nutrient_names)}"
    return env_id


def to_latex(x, precision=3, mathmode=True):
    mantissa, exp = f"{x:.{precision - 1}e}".split("e")
    exp = int(exp)
    s = f"{mantissa}\\times 10^{{{exp}}}"
    return f"${s}$" if mathmode else s


def load_ra_results(input_dir, ra_pattern="*_ra_results.csv", id_from_path=None):
    if id_from_path is None:
        id_from_path = lambda path: re.search(r"(\w+)_ra\w+", str(path)).group(1).capitalize()

    return [
        (id_from_path(ra_path), pd.read_csv(ra_path, index_col=0, dtype={"n": "int"}))
        for ra_path in sorted(Path(input_dir).glob(ra_pattern))
    ]


def calc_tradeoff_corr(df):
    if len(df) < 2 or df["delta_pf"].std(ddof=0) == 0 or df["delta_dt"].std(ddof=0) == 0:
        return 0
    return df["delta_pf"].corr(df["delta_dt"])


def calc_scores(ra_dfs, title, env_df=None, output_dir="output", n_mutations=5, k_min=1, same_desired_trait=True):
    max_dt = np.max([ra_df.loc["p_all", "desired_trait"] for _, ra_df in ra_dfs])

    sample_stat_results = []
    results = []
    for env_id, ra_df in ra_dfs:
        ra_df.sort_values("n", inplace=True)

        ss_df = ra_df.groupby("n")["proxy_fitness"].count().to_frame().T.rename(index={"proxy_fitness": env_id})
        ss_df = ss_df.loc[:, [col for col in ss_df.columns if col <= n_mutations]]
        ss_df = ss_df[[col for col in ss_df.columns if col >= 0] + [col for col in ss_df.columns if col < 0]]
        ss_df = ss_df.drop(columns=[0])
        ss_df.columns = [f"n=={int(col)}" for col in ss_df.columns]
        ss_df["total_count"] = ss_df.sum(axis=1)
        sample_stat_results.append(ss_df)

        if same_desired_trait:
            dt_baseline = max_dt
        else:
            dt_baseline = ra_df.loc["p_all", "desired_trait"]
        ra_df["norm_pf"] = ra_df["proxy_fitness"] / ra_df.loc["a_all", "proxy_fitness"]
        ra_df["norm_dt"] = ra_df["desired_trait"] / dt_baseline
        ra_df["delta_pf"] = ra_df["proxy_fitness"] - ra_df.loc["p_all", "proxy_fitness"]
        ra_df["delta_dt"] = ra_df["desired_trait"] - ra_df.loc["p_all", "desired_trait"]
        # ra_df["ddxdf"] = ra_df["delta_pf"] * ra_df["delta_dt"]
        valid_df = ra_df.loc[(~ra_df.index.isin(["p_all", "a_all"])) & ra_df["n"].between(k_min, n_mutations)]
        ra_df["ddxdf"] = np.nan
        for n_value in sorted(valid_df["n"].unique()):
            ra_df.loc[valid_df.index[valid_df["n"] == n_value], "ddxdf"] = calc_tradeoff_corr(valid_df[valid_df["n"] <= n_value])
        accepted_count = len(valid_df)
        if accepted_count == 0:
            robustness_score = 0
            tradeoff_score = 0
        else:
            robustness_score = valid_df["norm_dt"].mean()
            tradeoff_score = calc_tradeoff_corr(valid_df)
        results.append([env_id, accepted_count, robustness_score, tradeoff_score])

    sample_stat_res_df = pd.concat(sample_stat_results, axis=0).T
    sample_stat_res_df.to_csv(f"{output_dir}/{title.lower()}_sample_counts.csv")

    res_df = pd.DataFrame(
        results,
        columns=["env_id", "accepted_count", "robustness_score", "trade-off_score"],
    ).set_index("env_id")
    if env_df is not None:
        res_df = res_df.join(env_df["nutrients"], how="left")[["nutrients", "accepted_count", "robustness_score", "trade-off_score"]]
    res_df.to_csv(f"{output_dir}/{title.lower()}_score_results.csv")
    return res_df


def plot_line(
    title,
    output_dir,
    df_all,
    x,
    y,
    hue,
    xlabel,
    ylabel,
    legend,
    colors=plt.get_cmap("tab20").colors,
    show_legend=True,
    inset_y=None,
    inset_y_label=None,
):
    ax = plt.gca()
    axins = None
    if inset_y:
        axins = inset_axes(ax, width="40%", height="40%", loc="upper center")
        axins.tick_params(axis="both", which="both", labelsize=16)

    sns.lineplot(x=df_all[x].astype(int), y=y, hue=hue, data=df_all, ax=ax, errorbar=("ci", 95), palette=colors, linewidth=2)
    for c in ax.collections:
        if isinstance(c, mcoll.PolyCollection):
            c.set_alpha(0.1)
    if axins:
        g = sns.lineplot(
            x=df_all[x].astype(int),
            y=inset_y,
            hue=hue,
            data=df_all,
            ax=axins,
            errorbar=("ci", 95),
            palette=colors,
            linewidth=2,
            legend=False,
        )
        g.set_xlabel(None)
        g.set_ylabel(inset_y_label, fontsize=16)
        for c in axins.collections:
            if isinstance(c, mcoll.PolyCollection):
                c.set_alpha(0.1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend([mpatches.Patch(color=c) for c in colors], df_all[legend].drop_duplicates(), fontsize=16, loc="upper right")
    else:
        ax.legend().remove()
    if not os.path.exists(f"{output_dir}/figures/"):
        os.makedirs(f"{output_dir}/figures/")

    plt.tight_layout()
    output_path = Path(output_dir) / "figures" / f"{title.lower()}_{y}_lineplot.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    display(Image(filename=str(output_path), width=600))


def plot_score_results(
    title,
    output_dir,
    res_df,
    ra_dfs,
    env_list,
    env_df=None,
    n_mutations=5,
    k_min=1,
    colors=plt.get_cmap("tab20").colors,
):
    filtered_ra_dfs = [(to_env_name(env_id, env_df), ra_df[ra_df["n"].between(k_min, n_mutations)]) for env_id, ra_df in ra_dfs]
    df_all = pd.concat([ra_df.assign(env_name=env_name) for env_name, ra_df in filtered_ra_dfs], ignore_index=True)
    env_name_list = [to_env_name(env_id, env_df) for env_id, _ in res_df.loc[env_list, :].iterrows()]
    env_list, env_name_list = zip(*sorted(list(zip(env_list, env_name_list)), key=lambda x: x[1]))
    df_all = df_all[df_all["env_name"].isin(env_name_list)].sort_values("env_name")
    print(
        [
            f"{to_env_name(env_id, env_df)} ({env_id}) (RS = {to_latex(row['robustness_score'])}, TS = {to_latex(row['trade-off_score'])})"
            for env_id, row in res_df.loc[env_list, ["robustness_score", "trade-off_score"]].iterrows()
        ]
    )
    plot_line(
        title,
        output_dir,
        df_all,
        "n",
        "norm_dt",
        "env_name",
        "Number of changes in enzyme usage",
        "Desired trait (normalized)",
        "env_name",
        colors,
        True,
        "norm_pf",
        "Fitness (normalized)",
    )
    plot_line(
        title,
        output_dir,
        df_all,
        "n",
        "ddxdf",
        "env_name",
        "Number of changes in enzyme usage",
        r"Cumulative $\Delta f$-$\Delta d$ correlation",
        "env_name",
        colors,
        False,
    )


def show_score_results(
    title=None,
    input_dir=None,
    env_df=None,
    res_df=None,
    ra_dfs=None,
    env_list=None,
    n_mutations=5,
    k_min=1,
    same_desired_trait=True,
    colors=plt.get_cmap("tab20").colors,
):
    output_dir = os.path.dirname(input_dir) + "/"
    if title is None:
        title = os.path.basename(input_dir)
    if ra_dfs is None or res_df is None:
        ra_dfs = load_ra_results(input_dir)
        res_df = calc_scores(ra_dfs, title, env_df, output_dir, n_mutations, k_min, same_desired_trait)
    if env_list:
        plot_score_results(title, output_dir, res_df, ra_dfs, env_list, env_df, n_mutations, k_min, colors)
    return res_df, ra_dfs


def enumerate_env_list(res_df, ra_dfs, top_pct=0.5, random_state=0, n_envs=10):
    candidate_df = res_df.nlargest(max(n_envs, int(len(res_df) * top_pct)), "robustness_score").copy()
    dt_map = {env_id: ra_df.loc["p_all", "desired_trait"] for env_id, ra_df in ra_dfs if env_id in candidate_df.index}
    max_dt_env = max(dt_map, key=dt_map.get)
    remaining_df = candidate_df.drop(index=max_dt_env).sort_values("robustness_score").reset_index().rename(columns={"index": "env_id"})
    n_select = min(n_envs - 1, len(remaining_df))
    if n_select <= 0:
        return [max_dt_env]

    rng = np.random.default_rng(random_state)
    bin_edges = np.linspace(remaining_df["robustness_score"].min(), remaining_df["robustness_score"].max(), n_select + 1)
    selected_envs = []

    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == n_select - 1:
            pool = remaining_df[(remaining_df["robustness_score"] >= left) & (remaining_df["robustness_score"] <= right)]
        else:
            pool = remaining_df[(remaining_df["robustness_score"] >= left) & (remaining_df["robustness_score"] < right)]
        if pool.empty:
            dist = np.minimum(
                (remaining_df["robustness_score"] - left).abs(),
                (remaining_df["robustness_score"] - right).abs(),
            )
            pool = remaining_df[dist == dist.min()]
        chosen = pool.sample(n=1, random_state=int(rng.integers(0, 1_000_000_000))).iloc[0]
        selected_envs.append(chosen["env_id"])
        remaining_df = remaining_df[remaining_df["env_id"] != chosen["env_id"]]

    env_list = [max_dt_env] + selected_envs
    return sorted(env_list, key=lambda env_id: res_df.loc[env_id, "robustness_score"])
