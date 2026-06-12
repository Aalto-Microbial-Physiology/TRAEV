import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from adjustText import adjust_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from traev.constants import carbon_sources, substrate_symbols


def cluster_rs_ts_environments(res_df, n_clusters=4, random_state=0):
    df = res_df.copy().reset_index().rename(columns={"index": "env_id"})
    X = df[["robustness_score", "trade-off_score"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df["rs_ts_cluster"] = model.fit_predict(X_scaled)

    cluster_stats = df.groupby("rs_ts_cluster")[["robustness_score", "trade-off_score"]].agg(["count", "mean", "std"])
    cluster_stats = cluster_stats.sort_values(("robustness_score", "mean"), ascending=False)
    rs_mean = df["robustness_score"].mean()
    ts_mean = df["trade-off_score"].mean()
    return df, cluster_stats, rs_mean, ts_mean


def summarize_cluster_nutrients(cluster_df, env_df):
    merged = cluster_df.set_index("env_id").join(env_df[["nutrients"]], how="left")
    merged["nutrient_list"] = merged["nutrients"].apply(lambda x: [n.strip() for n in x.split(",")])

    all_envs = env_df.copy()
    all_envs["nutrient_list"] = all_envs["nutrients"].apply(lambda x: [n.strip() for n in x.split(",")])

    def count_features(series, size=1):
        counts = {}
        for items in series:
            if size == 1:
                features = items
            else:
                features = [
                    ", ".join(sorted(pair))
                    for pair in combinations(items, size)
                    if not all(substrate_symbols.get(n, n) in carbon_sources for n in pair)
                ]
            for feat in features:
                counts[feat] = counts.get(feat, 0) + 1
        return counts

    def feature_mask(series, feature, size):
        if size == 1:
            return series.apply(lambda items: feature in items)
        feature_parts = tuple(sorted(n.strip() for n in feature.split(",")))
        return series.apply(lambda items: all(part in items for part in feature_parts))

    def to_df(cluster_counts, all_counts, feature_type):
        rows = []
        n_envs = len(merged)
        total_envs = len(all_envs)
        size = 1 if feature_type == "single" else 2
        for feat, cnt in cluster_counts.items():
            cluster_freq = cnt / n_envs
            overall_freq = all_counts.get(feat, 0) / total_envs
            fold_enrichment = cluster_freq / overall_freq if overall_freq > 0 else np.nan
            score = np.log2(fold_enrichment) * np.sqrt(cnt) if pd.notna(fold_enrichment) and fold_enrichment > 0 else np.nan
            mask = feature_mask(merged["nutrient_list"], feat, size)
            rows.append(
                [
                    feature_type,
                    feat,
                    cnt,
                    cluster_freq,
                    overall_freq,
                    fold_enrichment,
                    score,
                    merged.loc[mask, "robustness_score"].mean(),
                    merged.loc[mask, "trade-off_score"].mean(),
                ]
            )
        out = pd.DataFrame(
            rows,
            columns=[
                "feature_type",
                "feature",
                "count",
                "cluster_freq",
                "overall_freq",
                "fold_enrichment",
                "score",
                "feature_mean_rs",
                "feature_mean_ts",
            ],
        )
        return out.sort_values(["score", "cluster_freq", "fold_enrichment", "count"], ascending=False).reset_index(drop=True)

    single_df = to_df(count_features(merged["nutrient_list"], size=1), count_features(all_envs["nutrient_list"], size=1), "single")
    pair_df = to_df(count_features(merged["nutrient_list"], size=2), count_features(all_envs["nutrient_list"], size=2), "pair")
    return single_df, pair_df


def plot_rs_ts_clusters(cluster_df, title, output_dir):
    fig, ax = plt.subplots()
    cluster_order = sorted(cluster_df["rs_ts_cluster"].unique())
    palette = {cluster_id: sns.color_palette("tab10")[int(cluster_id)] for cluster_id in cluster_order}
    sns.scatterplot(
        data=cluster_df,
        x="robustness_score",
        y="trade-off_score",
        hue="rs_ts_cluster",
        palette=palette,
        s=55,
        ax=ax,
        alpha=0.85,
        legend=False,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="w",
            label=str(cluster_id),
            markerfacecolor=palette[cluster_id],
            markeredgecolor=palette[cluster_id],
            markeredgewidth=0.0,
            markersize=8,
        )
        for cluster_id in cluster_order
    ]
    legend = ax.legend(handles, [h.get_label() for h in handles], frameon=True, title="Cluster", loc="upper right")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_alpha(0.9)
    ax.set_xlabel("Robustness Score")
    ax.set_ylabel("Trade-off Score")
    if not os.path.exists(f"{output_dir}/figures/"):
        os.makedirs(f"{output_dir}/figures/")
    plt.tight_layout()
    output_path = Path(output_dir) / "figures" / f"{title.lower()}_rs_ts_clusters.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    display(Image(filename=str(output_path), width=600))


def plot_feature_mean_rs_ts(feature_df, title, output_dir, abbr=True):
    if feature_df.empty:
        return

    fig, ax = plt.subplots()
    palette = dict(
        zip(
            sorted(feature_df["cluster"].unique()),
            sns.color_palette("tab10", n_colors=feature_df["cluster"].nunique()),
        )
    )

    single_highlight_df = feature_df[feature_df["feature_type"] == "single"].copy()
    if not single_highlight_df.empty:
        single_highlight_df["cluster_rank"] = single_highlight_df.groupby("feature")["score"].rank(method="first", ascending=False)
        single_highlight_df = single_highlight_df[(single_highlight_df["cluster_rank"] == 1) & (single_highlight_df["score"] > 0)]
        single_highlight_df = (
            single_highlight_df.sort_values(["cluster", "score", "cluster_freq", "count"], ascending=[True, False, False, False])
            .groupby("cluster", group_keys=False)
            .head(3)
        )

    pair_highlight_df = (
        feature_df[(feature_df["feature_type"] == "pair") & (feature_df["score"] > 0)]
        .sort_values(["cluster", "score", "cluster_freq", "count"], ascending=[True, False, False, False])
        .groupby("cluster", group_keys=False)
        .head(10)
    )

    highlight_df = pd.concat([single_highlight_df, pair_highlight_df], ignore_index=True)
    highlight_keys = set(zip(highlight_df["cluster"], highlight_df["feature_type"], highlight_df["feature"]))
    background_df = feature_df[
        [key not in highlight_keys for key in zip(feature_df["cluster"], feature_df["feature_type"], feature_df["feature"])]
    ].copy()

    if not background_df.empty:
        sns.scatterplot(
            data=background_df,
            x="feature_mean_rs",
            y="feature_mean_ts",
            size="cluster_freq",
            hue="cluster",
            palette=palette,
            sizes=(10, 200),
            alpha=0.4,
            legend=False,
            ax=ax,
        )

    if not highlight_df.empty:
        sns.scatterplot(
            data=highlight_df,
            x="feature_mean_rs",
            y="feature_mean_ts",
            size="cluster_freq",
            hue="cluster",
            palette=palette,
            sizes=(10, 200),
            alpha=1.0,
            legend=False,
            ax=ax,
        )

    texts = []
    for _, row in highlight_df.iterrows():
        label = ",".join(substrate_symbols[n.strip()] for n in row["feature"].split(",")) if abbr else row["feature"]
        fontsize = 10 if row["feature_type"] == "single" else 8
        texts.append(
            plt.text(
                row["feature_mean_rs"],
                row["feature_mean_ts"],
                label,
                fontsize=fontsize,
                weight="bold",
                alpha=1.0,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, edgecolor="gray"),
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, shrinkA=2, shrinkB=2),
            force_text=1.0,
            force_points=0.5,
            expand_points=(2.0, 2.5),
            expand_text=(2.0, 2.5),
            only_move={"points": "y", "text": "xy"},
            lim=1500,
        )

    plt.xlabel("Robustness Score")
    plt.ylabel("Trade-off Score")
    if not os.path.exists(f"{output_dir}/figures/"):
        os.makedirs(f"{output_dir}/figures/")
    plt.tight_layout()
    output_path = Path(output_dir) / "figures" / f"{title.lower()}_cluster_feature_mean_rs_ts.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    display(Image(filename=str(output_path), width=600))


def load_rs_ts_cluster_result(output_prefix):
    return {
        "cluster_df": pd.read_csv(f"{output_prefix}_cluster_assignments.csv"),
        "cluster_stats": pd.read_csv(f"{output_prefix}_cluster_stats.csv", header=[0, 1], index_col=0),
        "feature_df": pd.read_csv(f"{output_prefix}_cluster_features.csv"),
    }


def summarize_favorable_rs_ts_clusters(res_df, env_df, title, output_prefix, output_dir, n_clusters=4, random_state=0):
    cluster_df, cluster_stats, rs_mean, ts_mean = cluster_rs_ts_environments(
        res_df,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    plot_rs_ts_clusters(cluster_df, title, output_dir=output_dir)
    cluster_df.to_csv(f"{output_prefix}_cluster_assignments.csv", index=False)

    cluster_stats_to_save = cluster_stats.copy()
    cluster_stats_to_save[("global", "rs_mean")] = rs_mean
    cluster_stats_to_save[("global", "ts_mean")] = ts_mean
    cluster_stats_to_save.to_csv(f"{output_prefix}_cluster_stats.csv")

    rows_to_save = []
    for cluster_id in cluster_stats.index:
        sub = cluster_df[cluster_df["rs_ts_cluster"] == cluster_id][["env_id", "robustness_score", "trade-off_score"]]
        single_df, pair_df = summarize_cluster_nutrients(sub, env_df)
        combined_df = pd.concat([single_df, pair_df], axis=0, ignore_index=True)
        combined_df.insert(0, "cluster", cluster_id)
        combined_df.insert(1, "cluster_mean_rs", cluster_stats.loc[cluster_id, ("robustness_score", "mean")])
        combined_df.insert(2, "cluster_mean_ts", cluster_stats.loc[cluster_id, ("trade-off_score", "mean")])
        rows_to_save.append(combined_df)

    if rows_to_save:
        feature_df = pd.concat(rows_to_save, axis=0, ignore_index=True)
        feature_df.to_csv(f"{output_prefix}_cluster_features.csv", index=False)
        plot_feature_mean_rs_ts(feature_df, title, output_dir=output_dir)

    return load_rs_ts_cluster_result(output_prefix)
