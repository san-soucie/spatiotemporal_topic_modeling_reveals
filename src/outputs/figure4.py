import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import dvc.api

try:
    from .common import (
        project_dir,
        epoch1color,
        epoch2color,
        storm1color,
        storm2color,
        topic_colors,
    )
except ImportError:
    from common import (
        project_dir,
        epoch1color,
        epoch2color,
        storm1color,
        storm2color,
        topic_colors,
    )


def main():
    params = dvc.api.params_show()
    page_width = params["figures"]["page_width"]
    page_height = params["figures"]["page_height"]
    dpi = params["figures"]["dpi"]

    data_filename = project_dir / "data" / "model" / "rost_topic_prob.json"
    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    output_filename = project_dir / "output" / "figure4.eps"

    data = pd.read_json(data_filename).pivot(
        index="sample_time", columns="category", values="prob"
    )
    metadata = pd.read_json(metadata_filename)
    x = metadata.day_of_month
    y = data.to_numpy().T

    labels = [f"Community {i}" for i in range(1, 5)]
    fig, (tax, ax) = plt.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0},
        figsize=(page_width, page_height / 2),
    )

    ########
    # TIMELINE

    delta = (metadata.cruise_day - metadata.day_of_month).iloc[0]
    storm1start = 2 - delta
    storm1end = 6 - delta
    storm2start = 8.5 - delta
    storm2end = 11.2 - delta

    tax.axis("off")
    tax.set_xlim([min(x), max(x)])
    tax.set_ylim([0, 1])
    epoch1 = Rectangle(
        (min(x), 0.05),
        11 - min(x) - 0.025,
        0.95,
        linewidth=2,
        edgecolor="k",
        facecolor=epoch1color,
    )
    tax.add_patch(epoch1)
    tax.text(
        min(x) + (11 - min(x) - 0.025) / 2,
        0.65,
        "Epoch 1",
        horizontalalignment="center",
    )
    epoch2 = Rectangle(
        (11 + 0.025, 0.05),
        max(x) - (11) - 0.05,
        0.95,
        linewidth=2,
        edgecolor="k",
        facecolor=epoch2color,
    )
    tax.add_patch(epoch2)
    tax.text(
        11 + 0.025 + (max(x) - (11) - 0.05) / 2,
        0.65,
        "Epoch 2",
        horizontalalignment="center",
    )
    storm1 = Rectangle(
        (storm1start, 0.15),
        storm1end - storm1start,
        0.35,
        linewidth=2,
        edgecolor="k",
        facecolor=storm1color,
    )
    tax.add_patch(storm1)
    tax.text(
        storm1start + (storm1end - storm1start) / 2,
        0.25,
        "Storm 1",
        horizontalalignment="center",
    )
    storm2 = Rectangle(
        (storm2start, 0.15),
        storm2end - storm2start,
        0.35,
        linewidth=2,
        edgecolor="k",
        facecolor=storm2color,
    )
    tax.add_patch(storm2)
    tax.text(
        storm2start + (storm2end - storm2start) / 2,
        0.25,
        "Storm 2",
        horizontalalignment="center",
    )

    ########

    ax.stackplot(x, y, colors=topic_colors, labels=labels)
    legend = ax.legend()
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.6))

    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([0, 1])

    ax.set_xlabel("Day of month (May 2021)")
    ax.set_ylabel("Community proportion")
    fig.savefig(output_filename, dpi=dpi, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
