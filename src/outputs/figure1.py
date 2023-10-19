import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import dvc.api

try:
    from .common import project_dir, mark_text
except ImportError:
    from common import project_dir, mark_text


def main():
    params = dvc.api.params_show()
    page_width = params["figures"]["page_width"]
    page_height = params["figures"]["page_height"]
    dpi = params["figures"]["dpi"]

    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    output_filename = project_dir / "output" / "figure1.eps"
    ec_filename = project_dir / "data" / "raw" / "ec.csv"
    alt_filename = project_dir / "data" / "raw" / "altitude.csv"
    sst_filename = project_dir / "data" / "raw" / "sst.csv"

    fig, axes = plt.subplot_mosaic(
        "AB;CC", figsize=(page_width, page_height / 2), height_ratios=[4, 1]
    )
    ct_ax = axes["A"]
    sh_ax = axes["B"]
    kd_ax = axes["C"]

    ## CRUISE TRACK
    data = pd.read_json(metadata_filename)

    c = ct_ax.scatter(
        data.longitude,
        data.latitude,
        c=data.day_of_month,
        vmin=5,
        vmax=21,
        marker="$\u25EF$",
    )
    ct_cbar = plt.colorbar(c, label="Day of month (May 2021)", ax=ct_ax)
    ct_cbar.ax.set_yticks([5, 10, 15, 20])
    ct_ax.grid()
    ct_ax.set_xlim([-17, -13])
    ct_ax.set_xticks([-17, -16, -15, -14, -13])
    ct_ax.set_xlabel("Longitude")
    ct_ax.set_ylim([47, 51])
    ct_ax.set_yticks([47, 48, 49, 50, 51])
    ct_ax.set_ylabel("Latitude")

    rect = Rectangle((-16.5, 48), 3, 2, linewidth=3, edgecolor="k", facecolor="none")

    # Add the patch to the Axes
    ct_ax.add_patch(rect)

    ## SST/SSH MAP

    altitude = pd.read_csv(alt_filename)
    df = altitude.dropna()
    good_rows = df.longitude.between(-17, -13)
    df = df[good_rows]
    longitude = df.longitude.apply(float)
    good_columns = [
        c for c in df.columns if (c != "longitude") and (47.5 <= float(c) <= 50.5)
    ]
    df = df[good_columns]

    latitude = [float(x) for x in df.columns]
    x, y = np.meshgrid(longitude, latitude)
    z = df.to_numpy().transpose()

    sh_ax.contour(
        x,
        y,
        z,
        levels=np.linspace(-2, 2, int(4 / 0.02) + 1),
        colors="k",
        negative_linestyles="solid",
    )

    sst = pd.read_csv(sst_filename)
    df = sst.dropna(axis=1, how="all")
    good_rows = df.latitude.between(47.5, 50.5)
    df = df[good_rows]
    latitude = df.latitude.apply(float)
    good_columns = [
        c for c in df.columns if (c != "latitude") and (-17 <= float(c) <= -13)
    ]
    df = df[good_columns]
    longitude = [float(x) for x in df.columns]
    x, y = np.meshgrid(longitude, latitude)
    z = df.to_numpy()

    pcolor_img = sh_ax.pcolor(x, y, z, vmin=11.5, vmax=14, cmap="RdYlBu_r")
    ec = pd.read_csv(ec_filename, parse_dates=["sample_time"])
    sh_ax.scatter(
        ec.longitude[::2],
        ec.latitude[::2],
        marker="*",
        edgecolor="k",
        facecolor="grey",
        s=250,
    )
    sh_ax.scatter(
        ec.longitude[17],
        ec.latitude[17],
        marker="*",
        edgecolor="k",
        facecolor="y",
        s=250,
    )

    fig.colorbar(pcolor_img, label="SST ($\circ$C)", ax=sh_ax)
    sh_ax.set_xlim([-16.5, -13.5])
    sh_ax.set_xticks([-16, -15, -14])
    sh_ax.set_ylim([48, 50])
    sh_ax.set_yticks([48, 49, 50])
    sh_ax.set_xlabel("Longitude")
    sh_ax.set_ylabel("Latitude")

    metadata = pd.read_json(metadata_filename)

    delta = (metadata.cruise_day - metadata.day_of_month).iloc[0]
    storm1start = 2 - delta
    storm1end = 6 - delta
    storm2start = 8.5 - delta
    storm2end = 11.2 - delta

    kd_ax.set_xlim([0, 30])
    kd_ax.set_xticks([5, 10, 15, 20, 25])
    kd_ax.set_xticklabels([f"May {x}" for x in kd_ax.get_xticks()])

    kd_ax.set_ylim([0, 3])
    kd_ax.set_yticks([0, 1, 2, 3])
    kd_ax.set_yticklabels([""] * (len(kd_ax.get_xticks()) - 1))
    kd_ax.grid("on")

    rectangles = {
        "Epoch 1": Rectangle((0.1, 2.05), 9.8, 0.9, color="#a6bce3", label="Epoch 1"),
        "Epoch 2": Rectangle((10.1, 2.05), 9.8, 0.9, color="#ffe286", label="Epoch 2"),
        "Epoch 3": Rectangle((20.1, 2.05), 9.8, 0.9, color="#badba6", label="Epoch 3"),
        "Storm 1": Rectangle(
            (storm1start, 1.05),
            storm1end - storm1start,
            0.9,
            color="#f6c29d",
            label="Storm 1",
        ),
        "Storm 2": Rectangle(
            (storm2start, 1.05),
            storm2end - storm2start,
            0.9,
            color="#c1bdbc",
            label="Storm 2",
        ),
        "R/V $\it{Sarmiento}$ $\it{de}$ $\it{Gamboa}$ Operations": Rectangle(
            (min(metadata.day_of_month), 0.05),
            max(metadata.day_of_month) - min(metadata.day_of_month),
            0.9,
            color="#e3a9a8",
            label="Sarmiento",
        ),
    }
    for r in rectangles:
        kd_ax.add_patch(rectangles[r])
        rx, ry = rectangles[r].get_xy()
        cx = rx + rectangles[r].get_width() / 2.0
        cy = ry + rectangles[r].get_height() / 2.0

        kd_ax.annotate(r, (cx, cy), color="k", weight="bold", ha="center", va="center")

    mark_text(ct_ax, "a)", 0.033, 0.9)
    mark_text(sh_ax, "b)", 0.033, 0.9)
    kd_ax.text(0.5, 2.2, "c)", backgroundcolor="white")
    fig.tight_layout()

    fig.savefig(output_filename, dpi=dpi, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
