import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

from .scoring import calc_scores


def load_sensitivity(
    facet_value,
    base_dir,
    subdir_name,
    ra_pattern,
    id_from_path,
    same_desired_trait,
    facet_col,
    sort_cols,
    run_lookup=None,
    post_filter=None,
    should_include_run=None,
):
    def iter_runs():
        for run_dir in sorted(base_dir.glob("ra_*")):
            if run_lookup is None:
                params_path = run_dir / "params.json"
                params = json.loads(params_path.read_text(encoding="utf-8"))
            else:
                params = run_lookup[run_dir.name]
            yield run_dir, params

    def score_rows(score_df, params, run_id):
        return [
            {
                facet_col: facet_value,
                "run_id": run_id,
                "n_mutations": params["n_mutations"],
                "n_samples": params["n_samples"],
                "group_id": group_id,
                "accepted_count": row["accepted_count"],
                "robustness_score": row["robustness_score"],
                "tradeoff_score": row["trade-off_score"],
            }
            for group_id, row in score_df.iterrows()
        ]

    rows = []
    for run_dir, params in iter_runs():
        if should_include_run is not None and not should_include_run(params):
            continue
        data_dir = run_dir / subdir_name

        ra_dfs = [
            (id_from_path(ra_path), pd.read_csv(ra_path, index_col=0))
            for ra_path in sorted(data_dir.glob(ra_pattern))
        ]

        score_df = calc_scores(
            ra_dfs,
            facet_value,
            output_dir=run_dir,
            n_mutations=params["n_mutations"],
            same_desired_trait=same_desired_trait,
        )
        rows.extend(score_rows(score_df, params, run_dir.name))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
    return post_filter(summary_df) if post_filter is not None else summary_df


def plot_sensitivity_line(
    summary_df,
    metric,
    group_order,
    palette,
    output_path,
    group_col="group_id",
    legend_labels=None,
    legend_title=None,
    show_legend=True,
):
    ordered_df = summary_df.sort_values(["n_mutations", "n_samples"]).copy()
    ordered_df["param_label"] = ordered_df.apply(
        lambda row: f"({int(row['n_mutations'])}, {int(row['n_samples'])})",
        axis=1,
    )
    param_order = ordered_df["param_label"].drop_duplicates().tolist()
    ylabel = {"robustness_score": "Robustness score", "tradeoff_score": "Trade-off score"}[metric]
    ordered_df["param_label"] = pd.Categorical(ordered_df["param_label"], categories=param_order, ordered=True)
    ordered_df = ordered_df.sort_values(["param_label", group_col])

    ax = plt.gca()
    sns.lineplot(
        data=ordered_df,
        x="param_label",
        y=metric,
        hue=group_col,
        hue_order=group_order,
        palette=palette,
        marker="o",
        linewidth=2,
        errorbar=None,
        legend=show_legend,
        ax=ax,
    )
    ax.set_xlabel("(n_mutations, n_samples)")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    if show_legend:
        legend = ax.legend(
            handles,
            legend_labels if legend_labels is not None else labels,
            loc="upper right",
            title=legend_title,
            frameon=True,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("gray")
        legend.get_frame().set_alpha(0.9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    display(Image(filename=str(output_path), width=600))


def show_sensitivity_results(
    summary_df,
    output_dir,
    output_prefix,
    group_order=None,
    palette=None,
    group_col="group_id",
    legend_labels=None,
    legend_title=None,
):
    if group_order is None:
        group_order = sorted(summary_df[group_col].dropna().unique().tolist())
    if palette is None:
        colors = sns.color_palette("tab10", n_colors=len(group_order))
        palette = dict(zip(group_order, colors))

    output_dir = Path(output_dir)
    summary_df.to_csv(output_dir / f"{output_prefix}_sensitivity.csv", index=False)
    plot_sensitivity_line(
        summary_df,
        "robustness_score",
        group_order,
        palette,
        output_dir / "figures" / f"{output_prefix}_robustness_score_sensitivity_lineplot.png",
        group_col=group_col,
        legend_labels=legend_labels,
        legend_title=legend_title,
        show_legend=True,
    )
    plot_sensitivity_line(
        summary_df,
        "tradeoff_score",
        group_order,
        palette,
        output_dir / "figures" / f"{output_prefix}_tradeoff_score_sensitivity_lineplot.png",
        group_col=group_col,
        legend_labels=legend_labels,
        legend_title=legend_title,
        show_legend=False,
    )
    return summary_df
