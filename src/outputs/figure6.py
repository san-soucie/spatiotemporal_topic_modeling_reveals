import pandas as pd
import matplotlib.pyplot as plt
import dvc.api
import math
import numpy as np
import matplotlib

try:
    from .common import (
        project_dir,
        mark_text,
    )
except ImportError:
    from common import (
        project_dir,
        mark_text,
    )


def main():
    params = dvc.api.params_show()
    page_width = params["figures"]["page_width"]
    page_height = params["figures"]["page_height"]
    dpi = params["figures"]["dpi"]

    data_filename = project_dir / "data" / "model" / "rost_topic_prob.json"
    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    output_filename = project_dir / "output" / "figure6.eps"

    # topics over time space

    # source: https://gis.stackexchange.com/questions/372400/kilometer-to-degree-and-back
    earth_radius = 6271.0
    degrees_to_radians = math.pi / 180.0
    radians_to_degrees = 180.0 / math.pi

    def change_in_latitude(kms):
        "Given a distance north, return the change in latitude."
        return (kms / earth_radius) * radians_to_degrees

    def change_in_longitude(latitude, kms):
        "Given a latitude and a distance west, return the change in longitude."
        # Find the radius of a circle around the earth at given latitude.
        r = earth_radius * math.cos(latitude * degrees_to_radians)
        return (kms / r) * radians_to_degrees

    data = pd.read_json(data_filename).pivot(
        index="sample_time", columns="category", values="prob"
    )
    data = data.rename({f"topic_{i}": f"Community {i+1}" for i in range(5)}, axis=1)

    metadata = pd.read_json(metadata_filename)

    def cruise_period_from_day(t):
        if t < 2:
            return "before_storm1"
        elif t < 6:
            return "storm1"
        elif t < 8.5:
            return "between_storms"
        elif t < 11.2:
            return "storm2"
        else:
            return "after_storm2"

    metadata["cruise_period"] = metadata.cruise_day.apply(cruise_period_from_day)
    eddy_centroids = metadata.groupby("cruise_period")[["lonec", "latec"]].apply("mean")

    fig, axes = plt.subplots(4, 5, figsize=(page_width, page_height))
    communities = list(data.columns)
    fancy_communities = [f"Community {i}\nLatitude" for i in range(1, 6)]
    cruise_periods = [
        "before_storm1",
        "storm1",
        "between_storms",
        "storm2",
        "after_storm2",
    ]
    fancy_cruise_periods = [
        "Longitude\nBefore storm 1\nMay 5-7",
        "Longitude\nStorm 1\nMay 7-11",
        "Longitude\nBetween storms\nMay 11-14",
        "Longitude\nStorm 2\nMay 14-16",
        "Longitude\nAfter storm 2\nMay 16-19",
    ]

    delta_lat = change_in_latitude(15)
    delta_lon = [change_in_longitude(x, 15) for x in eddy_centroids.latec]
    degrees = np.linspace(0, 2 * math.pi, 100)
    labels = [f"{x})" for x in "abcdefghijklmnopqrstuvwxyz"]

    for x in range(4):
        for y in range(5):
            ax = axes[x, y]
            c = communities[x]
            t = cruise_periods[y]
            tf = fancy_cruise_periods[y]
            i = list(metadata.cruise_period == t)
            xec = eddy_centroids.at[t, "lonec"]
            yec = eddy_centroids.at[t, "latec"]
            dx = delta_lon[y]
            dy = delta_lat
            storm = t in ("storm1", "storm2")
            ax.plot(
                [xec],
                [yec],
                marker="x",
                markersize=2 if storm else 5,
                markeredgecolor="red",
                markerfacecolor="red",
                alpha=0.5,
            )
            ax.plot(
                xec + dx * np.cos(degrees),
                yec + dy * np.sin(degrees),
                color="red",
                alpha=0.5,
            )
            scatter = ax.scatter(
                metadata.longitude[i],
                metadata.latitude[i],
                c=data[c][i],
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            storm = t in ["storm1", "storm2"]
            if storm:
                ax.set_facecolor("grey")
            else:
                ax.set_xlim([-15.3, -14.4])
                ax.set_ylim([48.5, 49.3])
            ax.grid()
            if x == 3:
                ax.set_xlabel(tf)
            if y == 0:
                ax.set_ylabel(fancy_communities[x])
            ax.tick_params(axis="both", which="major", labelsize=8)
            mark_text(ax, labels[x * 5 + y], dx=0.3 / 3, dy=(2 - 0.275) / 2)

    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.025, 0.7])
    legend_ax = fig.add_axes([0.86, 0.88, 0.025, 0.1])
    lelts1 = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            color="w",
            marker="$\odot$",
            markerfacecolor="r",
            markersize=10,
            markeredgecolor="r",
            label="Eddy",
            alpha=0.5,
        ),
    ]
    fig.colorbar(scatter, cax=cbar_ax, label="Community proportion")
    legend_ax.axis("off")
    legend_ax.legend(handles=lelts1, loc="center")
    fig.savefig(output_filename, dpi=dpi, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
