from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import Image, display
from adjustText import adjust_text
from scipy.stats import pearsonr

from traev.constants import substrate_symbols


def calculate_nutrient_effect(df, title, output_dir, pc1="robustness_score", pc2="trade-off_score"):
    df = df.reset_index()
    df["nutrients"] = df["nutrients"].apply(lambda x: [n.strip() for n in x.split(",")])

    y1 = (df[pc1] - df[pc1].mean()) / df[pc1].std()
    y2 = (df[pc2] - df[pc2].mean()) / df[pc2].std()

    all_nutrients = sorted(set(n for row in df["nutrients"] for n in row))
    nutrient_matrix = pd.DataFrame(df["nutrients"].apply(lambda ns: {n: 1 for n in ns}).tolist()).fillna(0).astype(int)
    nutrient_matrix = nutrient_matrix.reindex(columns=all_nutrients, fill_value=0)
    df = pd.concat([df, nutrient_matrix], axis=1)

    X_single = sm.add_constant(df[all_nutrients])
    model_r_single = sm.OLS(y1, X_single).fit()
    model_e_single = sm.OLS(y2, X_single).fit()

    res_single = pd.DataFrame(
        {
            "nutrients": all_nutrients,
            "n_nutrients": [1] * len(all_nutrients),
            "envs": [df[df[n] == 1]["env_id"].tolist() for n in all_nutrients],
            f"{pc1}_wt": model_r_single.params.values[1:],
            f"{pc1}_pv": model_r_single.pvalues.values[1:],
            f"{pc2}_wt": model_e_single.params.values[1:],
            f"{pc2}_pv": model_e_single.pvalues.values[1:],
        }
    )

    df["nutrient_pairs"] = df["nutrients"].apply(lambda x: [", ".join(sorted(pair)) for pair in combinations(x, 2)])
    all_pairs = sorted(set(pair for row in df["nutrient_pairs"] for pair in row))
    pair_matrix = pd.DataFrame(df["nutrient_pairs"].apply(lambda pairs: {pair: 1 for pair in pairs}).tolist()).fillna(0).astype(int)
    pair_matrix = pair_matrix.reindex(columns=all_pairs, fill_value=0)
    df = pd.concat([df, pair_matrix], axis=1)

    X_pairs = sm.add_constant(df[all_pairs])
    model_r_pair = sm.OLS(y1, X_pairs).fit()
    model_e_pair = sm.OLS(y2, X_pairs).fit()

    res_pair = pd.DataFrame(
        {
            "nutrients": all_pairs,
            "n_nutrients": [2] * len(all_pairs),
            "envs": [df[df[p] == 1]["env_id"].tolist() for p in all_pairs],
            f"{pc1}_wt": model_r_pair.params.values[1:],
            f"{pc1}_pv": model_r_pair.pvalues.values[1:],
            f"{pc2}_wt": model_e_pair.params.values[1:],
            f"{pc2}_pv": model_e_pair.pvalues.values[1:],
        }
    )

    res_df = pd.concat([res_single, res_pair], ignore_index=True).sort_values(by=["n_nutrients", "nutrients"], ascending=[True, True]).reset_index(drop=True)
    res_df.to_csv(f"{output_dir}/{title.lower()}_nut_effect_on_{pc1}_{pc2}.csv", index=False)
    return res_df


def plot_nut_scatter(nut_df, n_nutrients, title, output_dir, pv=0.1, n_envs=5, abbr=True, pc1="robustness_score", pc2="trade-off_score"):
    nut_df = nut_df[nut_df["n_nutrients"] == n_nutrients]
    nut_df = nut_df[nut_df["envs"].apply(lambda x: len(x) >= n_envs)]
    nut_df = nut_df.sort_values(by=f"{pc1}_wt", ascending=False)
    r, p = pearsonr(nut_df[f"{pc1}_wt"], nut_df[f"{pc2}_wt"])

    mask = (nut_df[f"{pc1}_pv"] <= pv) & (nut_df[f"{pc2}_pv"] <= pv)
    nut_df["is_significant"] = mask

    ax = plt.gca()
    color_palette = dict(
        zip(
            nut_df["nutrients"],
            matplotlib.colormaps["Spectral"].resampled(len(nut_df))(range(len(nut_df))),
        )
    )

    sns.scatterplot(
        data=nut_df[~nut_df["is_significant"]],
        x=f"{pc1}_wt",
        y=f"{pc2}_wt",
        size=nut_df[~nut_df["is_significant"]]["envs"].apply(len),
        hue="nutrients",
        palette=color_palette,
        sizes=(10, 200),
        alpha=0.4,
        legend=False,
        ax=ax,
    )

    sns.scatterplot(
        data=nut_df[nut_df["is_significant"]],
        x=f"{pc1}_wt",
        y=f"{pc2}_wt",
        size=nut_df[nut_df["is_significant"]]["envs"].apply(len),
        hue="nutrients",
        palette=color_palette,
        sizes=(10, 200),
        alpha=1.0,
        legend=False,
        ax=ax,
    )

    ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
    ax.axvline(0, linestyle="--", color="gray", alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, title=f"r = {r:.3f}", title_fontsize=16)
    print(f"r = {r:.3f}, p = {p:.3f}")

    texts = []
    for _, row in nut_df[(nut_df["is_significant"]) | (nut_df["n_nutrients"] == 1)].iterrows():
        label = ",".join(substrate_symbols[n.strip()] if abbr else n.strip().capitalize() for n in row["nutrients"].split(","))

        fontsize = 10 if row["n_nutrients"] == 1 else 8
        weight = "bold"
        alpha = 1.0 if row["is_significant"] else 0.6

        texts.append(
            plt.text(
                row[f"{pc1}_wt"],
                row[f"{pc2}_wt"],
                label,
                fontsize=fontsize,
                weight=weight,
                alpha=alpha,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, edgecolor="gray"),
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, shrinkA=2, shrinkB=2),
        force_text=1.0,
        force_points=0.5,
        expand_points=(2.0, 2.5),
        expand_text=(2.0, 2.5),
        only_move={"points": "y", "text": "xy"},
        lim=1500,
    )

    plt.xlabel(f"Effect on {' '.join([s[0].upper() + s[1:] for s in pc1.split('_')])}")
    plt.ylabel(f"Effect on {' '.join([s[0].upper() + s[1:] for s in pc2.split('_')])}")
    plt.tight_layout()

    output_path = Path(output_dir) / "figures" / f"{title.lower()}_nut_effect_scatterplot_on_{pc1}_{pc2}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    display(Image(filename=str(output_path), width=600))
