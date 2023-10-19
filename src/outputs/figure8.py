import pandas as pd
import matplotlib.pyplot as plt
import dvc.api
import numpy as np
from scipy.stats import entropy
import scipy
import matplotlib

try:
    from .common import project_dir, mark_text
except ImportError:
    from common import project_dir, mark_text


def datetime_to_day_of_month(t):
    return t.day + t.hour / 24 + t.minute / (60 * 24) + t.second / (60 * 60 * 24)


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
    data_filename = project_dir / "data" / "processed" / "data.json"
    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    raw_classcount_filename = (
        project_dir / "data" / "raw" / "summary_biovol_allHDF_min20_2021.classcount.csv"
    )
    raw_metadata_filename = (
        project_dir / "data" / "raw" / "summary_biovol_allHDF_min20_2021.meta_data.csv"
    )
    raw_1dmodel_filename = project_dir / "data" / "raw" / "1dmodel.csv"
    output_filename = project_dir / "output" / "figure8.eps"

    fig = plt.figure(figsize=(page_width, page_height / 2))

    ## data
    data = pd.read_json(metadata_filename)

    fig = plt.figure(figsize=(page_width, page_height / 2))

    ax = plt.subplot(1, 7, (1, 3))
    data = data[data.watermass == "core"]
    x = [7, 13.5, 18.2]
    bounds = [(5, 9), (9, 16), (16, 21)]
    labels = ["Before storm 1", "Between storms", "After storm 2"]
    for b, lab in zip(bounds, labels):
        idx = data.day_of_month.between(*b)
        ax.scatter(data[idx].day_of_month, data[idx].fluorescence, label=lab, s=100)
    y = [data[data.day_of_month.between(*b)].fluorescence.mean() for b in bounds]
    e = [data[data.day_of_month.between(*b)].fluorescence.std() for b in bounds]
    ax.errorbar(
        x, y, e, linestyle="None", marker="^", c="k", markersize=20, linewidth=5
    )
    plt.grid("on")
    linreg = scipy.stats.linregress(data.day_of_month, data.fluorescence)
    linreg.rvalue
    ax.plot(
        data.day_of_month,
        np.array(data.day_of_month) * float(linreg.slope) + linreg.intercept,
        c="k",
        linewidth=5,
        label=f"Best-fit line $(r^2={linreg.rvalue**2:3.3f})$\n$p={linreg.pvalue:.2e}$",
    )
    ax.set_ylim([0, 3])
    plt.legend(loc="upper right")
    ax.set_xlabel("Day of month (May 2021)")
    ax.set_ylabel("Chlorophyll fluorescence")

    mark_text(ax, "a)", dx=0.05, dy=0.95)
    first_ax = ax
    # 2

    data = pd.read_csv(raw_classcount_filename)
    metadata = pd.read_csv(raw_metadata_filename, parse_dates=["sample_time"])
    data = data.div(data.sum(axis=1), axis=0)
    idx = metadata.sample_type == "cast"
    data = data[idx]
    metadata = metadata[idx]
    metadata_rec = pd.read_json(metadata_filename)
    r = metadata_rec.r_ec
    data2 = pd.read_json(data_filename)
    data2 = data2.div(data2.sum(axis=1), axis=0)

    model = pd.read_csv(raw_1dmodel_filename, parse_dates=["datetime"])
    model["day_of_month"] = [datetime_to_day_of_month(t) for t in model.datetime]
    model = model[model.datetime.apply(lambda t: t.month) == 5]
    model = model.dropna()

    y = data[
        "Pseudo-nitzschia"
    ]  # + min(x for x in data['Pseudo-nitzschia'] if x > 0) / 100000
    x = [datetime_to_day_of_month(t) for t in metadata.sample_time]

    z = metadata.depth

    x2 = [datetime_to_day_of_month(t) for t in metadata_rec.sample_time]
    y2 = data2["Pseudo-nitzschia"]

    ax = plt.subplot(2, 7, (11, 13))
    ax2 = plt.subplot(2, 7, (4, 6))
    ax.plot(
        model.day_of_month,
        model[["k-epsilon", "ePBL", "CVmix"]].sum(axis=1) * -1 / 3,
        linewidth=3,
    )
    im = ax.scatter(
        x, z, c=y, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1.0), s=100
    )
    ax.invert_yaxis()
    ax.grid("on")
    ax.set_yscale("log")
    ax.set_ylabel("Depth (m)")
    # ax.set_title('CTD Cast')
    ax.set_xlabel("Day of month (May 2021)")
    # (ax.twinx()).scatter(tr, r, c=r < 15)
    # (ax.twinx()).scatter(x2, y2, c=y2, vmin=0, vmax=1)
    ax2.scatter(x2, r, c=y2, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1.0), s=100)
    # fig.colorbar(im)
    ax2.grid("on")
    ax.set_xlim([min(min(x), min(x2)), max(max(x), max(x2))])
    ax2.set_xlim([min(min(x), min(x2)), max(max(x), max(x2))])
    ax2.set_yscale("log")
    ax2.set_ylabel("Distance (km)\n(to eddy center)")
    # ax2.set_title('Surface')
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(["" for _ in ax2.get_xticks()])
    fig.subplots_adjust(right=0.8)
    cbar_ax = plt.subplot(1, 7, 7)
    fig.colorbar(im, cax=cbar_ax, label="Pseudo-nitzschia rel. ab.")

    minx, maxx = ax2.get_xlim()
    x = minx + 0.05 * (maxx - minx)
    ax2.text(x, 150, "b)", backgroundcolor="white")
    minx, maxx = ax.get_xlim()
    x = minx + 0.05 * (maxx - minx)
    ax.text(x, 2.5, "c)", backgroundcolor="white")

    axes = [first_ax, ax, ax2]
    minx = min([a.get_xlim()[0] for a in axes])
    maxx = max([a.get_xlim()[1] for a in axes])
    for a in axes:
        a.set_xlim([minx, maxx])
        a.set_xticks([5, 10, 15, 20])

    # set_font_size(ax, size=FONTSIZE*1.5)
    # set_font_size(ax2, size=FONTSIZE*1.5)
    # set_font_size(cbar_ax, size=FONTSIZE*1.5)

    axes = [first_ax, ax, ax2]
    minx = min([a.get_xlim()[0] for a in axes])
    maxx = max([a.get_xlim()[1] for a in axes])
    for a in axes:
        a.set_xlim([minx, maxx])
        a.set_xticks([5, 10, 15, 20])
    ##
    fig.tight_layout()
    fig.savefig(output_filename, dpi=dpi, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
