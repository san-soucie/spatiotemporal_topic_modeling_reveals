import pandas as pd
import matplotlib.pyplot as plt
import dvc.api
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression


try:
    from .common import project_dir, topic_colors, mark_text
except ImportError:
    from common import project_dir, topic_colors, mark_text


def normalize(df):
    return df.div(df.sum(axis=1), axis=0)


def get_mixing(
    eddy_epoch1_means, filament_epoch1_means, cold_fresh_epoch1_means, eddy_epoch2
):
    def calc_kl_div(x, y):
        v = (
            x * eddy_epoch1_means
            + y * filament_epoch1_means
            + (1 - x - y) * cold_fresh_epoch1_means
        )
        return entropy(
            eddy_epoch2.to_numpy().T,
            v.to_numpy()[..., np.newaxis],
        )

    n = len(eddy_epoch2)
    df = pd.DataFrame({"x": [0.0] * n, "y": [0.0] * n, "dkl": [np.inf] * n})
    for x in np.linspace(0, 1.0, 101):
        for y in np.linspace(0, 1.0, 101):
            if x + y > 1:
                continue
            for i, d in enumerate(calc_kl_div(x, y)):
                if df.at[i, "dkl"] > d:
                    df.iloc[i] = [x, y, d]
    return df


def main():
    params = dvc.api.params_show()
    page_width = params["figures"]["page_width"]
    page_height = params["figures"]["page_height"]
    dpi = params["figures"]["dpi"]

    obs_filename = project_dir / "data" / "processed" / "data.json"
    data_filename = project_dir / "data" / "model" / "rost_topic_prob.json"
    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    output_filename = project_dir / "output" / "figure7.eps"

    fig = plt.figure(figsize=(page_width, page_height / 2))

    ## data
    observations = pd.read_json(obs_filename)
    metadata = pd.read_json(metadata_filename)
    topics = pd.read_json(data_filename).pivot(
        index="sample_time", columns="category", values="prob"
    )

    f1idx = list((metadata.cruise_day < 1.9) & (metadata.watermass == "warm_salty"))
    e1idx = list((metadata.cruise_day < 1.9) & (metadata.watermass == "core"))
    c1idx = list((metadata.cruise_day < 1.9) & (metadata.watermass == "cold_fresh"))
    e2idx = list(
        (metadata.cruise_day < 8.13)
        & (7.13 < metadata.cruise_day)
        & (metadata.watermass == "core")
    )

    filament_epoch1_obs_means = normalize(observations)[f1idx].mean()
    eddy_epoch1_obs_means = normalize(observations)[e1idx].mean()
    cold_fresh_epoch1_obs_means = normalize(observations)[c1idx].mean()
    eddy_epoch2_obs = normalize(observations)[e2idx]
    obs_mixing = get_mixing(
        eddy_epoch1_obs_means,
        filament_epoch1_obs_means,
        cold_fresh_epoch1_obs_means,
        eddy_epoch2_obs,
    )
    gidx = obs_mixing.dkl < np.inf
    obs_mixing = obs_mixing[gidx]
    e2eddy_metadata = metadata[e2idx].reset_index()[gidx]
    topics_e2idx = topics[e2idx].reset_index().drop("sample_time", axis=1)[gidx]

    ## a)

    axes = [
        plt.subplot(2, 7, (1, 2)),
        plt.subplot(2, 7, (3, 4)),
        plt.subplot(2, 7, (5, 6)),
    ]

    x = metadata.day_of_month

    x1 = x[f1idx]
    y1 = topics[f1idx].to_numpy().T

    x2 = x[e1idx]
    y2 = topics[e1idx].to_numpy().T

    x3 = x[e2idx]
    y3 = topics[e2idx].to_numpy().T

    colors = topic_colors
    labels = [f"Community {i}" for i in range(1, 6)]

    axes[0].stackplot(x1, y1, colors=colors, labels=labels)

    axes[0].set_xlim([min(x1), max(x1)])
    axes[0].set_ylim([0, 1])

    axes[0].set_ylabel("Community prop.")
    # axes[0].set_title('Filament\n(Epoch 1)')

    axes[1].stackplot(x2, y2, colors=colors, labels=labels)

    axes[1].set_xlim([min(x2), max(x2)])
    axes[1].set_ylim([0, 1])
    axes[1].set_yticklabels([])
    # axes[1].set_title('Eddy\n(Epoch 1)')

    # axes[1].set_xlabel('Day of month')

    axes[2].stackplot(x3, y3, colors=colors, labels=labels)

    axes[2].set_xlim([min(x3), max(x3)])
    # axes[2].set_xticklabels(["", "13", "13.5"])
    axes[2].set_ylim([0, 1])
    axes[2].set_yticklabels([])
    # axes[2].set_title('Eddy\n(Epoch 2)')

    axes[0].set_xlabel("Filament\n(Epoch 1)")
    axes[1].set_xlabel("Eddy\n(Epoch 1)\nDay of month")
    axes[2].set_xlabel("Eddy\n(Epoch 2)")

    # for ax in axes:
    #     set_font_size(ax, size=FONTSIZE*1.6)

    mark_text(axes[0], "a)", dx=0.25 / 3, dy=(2 - 0.25) / 2)
    mark_text(axes[1], "b)", dx=0.25 / 3, dy=(2 - 0.25) / 2)
    mark_text(axes[2], "c)", dx=0.25 / 3, dy=(2 - 0.25) / 2)

    ## b)

    ax = plt.subplot(2, 7, (8, 9))

    line = ([0, 1], [1, 0])
    ax.plot(*line, c="k", linewidth=3)
    im = ax.scatter(obs_mixing.x, obs_mixing.y, c=obs_mixing.dkl, s=100)
    ax.grid()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Eddy fraction")
    ax.set_ylabel("Filament fraction")
    # cbar = fig.colorbar(im, label='KL Divergence')
    # set_font_size(ax, size=FONTSIZE*1.6)
    # set_font_size(cbar.ax, size=FONTSIZE*1.6)
    for i, (xa, ya, ca) in enumerate(zip(obs_mixing.x, obs_mixing.y, obs_mixing.dkl)):
        if ya > 0.8:
            ax.scatter([xa], [ya], c=[ca], edgecolor="r", s=100)

    dx = 0.9 / 3
    dy = (2 - 0.25) / 2
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ax.text(
        minx + dx * (maxx - minx),
        miny + dy * (maxy - miny),
        "d)",
        backgroundcolor="white",
    )

    ## c)

    ax = plt.subplot(2, 7, (10, 11))

    x = e2eddy_metadata.longitude
    y = e2eddy_metadata.latitude
    c = obs_mixing.dkl

    im = ax.scatter(x, y, c=c, s=100)
    ax.grid()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    for i, (xa, ya, ca) in enumerate(zip(x, y, c)):
        if obs_mixing.at[i, "y"] > 0.8:
            ax.scatter([xa], [ya], c=[ca], edgecolor="r", s=100)
    # set_font_size(ax, size=FONTSIZE*1.6)
    # set_font_size(cbar.ax, size=FONTSIZE*1.6)
    ax.set_xticks([-14.9, -14.8, -14.7])
    ax.set_yticks([49.05, 49.1, 49.15])

    dx = 0.25 / 3
    dy = (2 - 0.6) / 2
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ax.text(
        minx + dx * (maxx - minx),
        miny + dy * (maxy - miny),
        "e)",
        backgroundcolor="white",
    )
    ## d)

    ax = plt.subplot(2, 7, (12, 13))

    x = obs_mixing.y
    print(topics_e2idx)
    y = topics_e2idx["topic_3"]
    c = obs_mixing.dkl

    linreg = LinearRegression().fit(x.to_numpy().reshape(-1, 1), y)
    m = linreg.coef_.item()
    b = linreg.intercept_
    r2 = linreg.score(x.to_numpy().reshape(-1, 1), y)

    line_x = [0.0, 1.0]
    line_y = [b, m + b]

    im = ax.scatter(x, y, c=c, s=100)
    ax.plot(line_x, line_y, c="k", linewidth=3)
    ax.text(0.1, 0.7, f"$r^2={r2:1.2f}$")
    ax.grid()
    ax.set_xlabel("Filament fraction")
    ax.set_ylabel("Community 3 proportion")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # set_font_size(ax, size=FONTSIZE*1.6)
    # set_font_size(cbar.ax, size=FONTSIZE*1.6)

    for i, (xa, ya, ca) in enumerate(zip(x, y, c)):
        if xa > 0.8:
            ax.scatter([xa], [ya], c=[ca], edgecolor="r", s=100)

    dx = 0.25 / 3
    dy = (2 - 0.25) / 2
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ax.text(
        minx + dx * (maxx - minx),
        miny + dy * (maxy - miny),
        "f)",
        backgroundcolor="white",
    )

    ## colorbar

    ax = plt.subplot(1, 7, 7)
    fig.colorbar(im, orientation="vertical", label="KL Divergence", cax=ax)

    ##
    plt.tight_layout()
    fig.savefig(output_filename, dpi=dpi, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
