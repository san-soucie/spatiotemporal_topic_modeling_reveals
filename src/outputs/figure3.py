import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import dvc.api
from .common import project_dir, epoch1color, epoch2color, storm1color, mark_text


def main():
    params = dvc.api.params_show()
    page_width = params["figures"]["page_width"]
    page_height = params["figures"]["page_height"]
    dpi = params["figures"]["dpi"]

    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    nmds_filename = project_dir / "data" / "model" / "nmds.json"
    output_filename = project_dir / "output" / "figure3.eps"

    fig = plt.figure(figsize=(page_width, page_height / 2))

    metadata = pd.read_json(metadata_filename)

    def cruise_period_from_day(t):
        if t < 2:
            return "before_storm1"
        elif t < 6:
            return "storm1"
        elif t < 9:
            return "between_storms"
        elif t < 11:
            return "storm2"
        else:
            return "after_storm2"

    metadata["cruise_period"] = metadata.cruise_day.apply(cruise_period_from_day)

    nmds = pd.read_json(nmds_filename)
    x = nmds["nmds_component_1"]
    y = nmds["nmds_component_2"]
    idx = (
        metadata.cruise_period.isin(("before_storm1", "between_storms"))
        & metadata.watermass.eq("core")
    ) | (
        metadata.cruise_period.eq("before_storm1") & metadata.watermass.eq("warm_salty")
    )
    x1 = x[idx]
    y1 = y[idx]
    m1 = metadata[idx]
    filament_epoch1_idx = m1.watermass == "warm_salty"
    core_epoch1_idx = (m1.watermass == "core") & (m1.cruise_period == "before_storm1")
    core_epoch2_idx = (m1.watermass == "core") & (m1.cruise_period == "between_storms")

    x1_filament_epoch1_mean = np.mean(x1[filament_epoch1_idx])
    y1_filament_epoch1_mean = np.mean(y1[filament_epoch1_idx])

    x1_core_epoch1_mean = np.mean(x1[core_epoch1_idx])
    y1_core_epoch1_mean = np.mean(y1[core_epoch1_idx])

    x1_core_epoch2_mean = np.mean(x1[core_epoch2_idx])
    y1_core_epoch2_mean = np.mean(y1[core_epoch2_idx])

    t2 = [
        storm1color
        if m1.loc[i, "watermass"] == "warm_salty"
        else epoch1color
        if m1.loc[i, "cruise_period"] == "before_storm1"
        else epoch2color
        for i in m1.index
    ]

    ax = plt.subplot(1, 2, 1)

    ax.grid()

    ax.set_axisbelow(True)
    ax.scatter(x1, y1, c=t2, vmin=5, vmax=21, edgecolors="k", s=100)
    ax.scatter(
        [x1_filament_epoch1_mean],
        [y1_filament_epoch1_mean],
        c=storm1color,
        marker="*",
        s=500,
        edgecolors="k",
        linewidths=2,
    )
    ax.scatter(
        [x1_core_epoch1_mean],
        [y1_core_epoch1_mean],
        c=epoch1color,
        marker="*",
        s=500,
        edgecolors="k",
        linewidths=2,
    )
    ax.scatter(
        [x1_core_epoch2_mean],
        [y1_core_epoch2_mean],
        c=epoch2color,
        marker="*",
        s=500,
        edgecolors="k",
        linewidths=2,
    )

    xa = x1_core_epoch1_mean
    ya = y1_core_epoch1_mean
    dx = x1_core_epoch2_mean - x1_core_epoch1_mean
    dy = y1_core_epoch2_mean - y1_core_epoch1_mean

    xa += 0.1 * dx
    ya += 0.1 * dy

    dx *= 0.8
    dy *= 0.8
    ax.arrow(
        xa,
        ya,
        dx,
        dy,
        width=0.06,
        length_includes_head=True,
        head_width=0.14,
        head_length=0.1,
        fill=True,
        facecolor="grey",
        edgecolor="k",
    )

    ax.set_xlabel("NMDS component 1")
    ax.set_ylabel("NMDS component 2")

    mark_text(ax, "a)", 0.05, 0.95)

    colors = [storm1color, epoch1color, epoch2color]
    idx = (
        metadata.cruise_period.isin(("before_storm1", "between_storms"))
        & metadata.watermass.eq("core")
    ) | (
        metadata.cruise_period.eq("before_storm1") & metadata.watermass.eq("warm_salty")
    )
    x1 = metadata.salinity[idx]
    y1 = metadata.temperature[idx]
    m1 = metadata[idx]
    filament_epoch1_idx = m1.watermass == "warm_salty"
    core_epoch1_idx = (m1.watermass == "core") & (m1.cruise_period == "before_storm1")
    core_epoch2_idx = (m1.watermass == "core") & (m1.cruise_period == "between_storms")

    x1_filament_epoch1_mean = np.mean(x1[filament_epoch1_idx])
    y1_filament_epoch1_mean = np.mean(y1[filament_epoch1_idx])

    x1_core_epoch1_mean = np.mean(x1[core_epoch1_idx])
    y1_core_epoch1_mean = np.mean(y1[core_epoch1_idx])

    x1_core_epoch2_mean = np.mean(x1[core_epoch2_idx])
    y1_core_epoch2_mean = np.mean(y1[core_epoch2_idx])

    t2 = [
        storm1color
        if m1.loc[i, "watermass"] == "warm_salty"
        else epoch1color
        if m1.loc[i, "cruise_period"] == "before_storm1"
        else epoch2color
        for i in m1.index
    ]
    ax = plt.subplot(1, 2, 2)
    ax.grid()
    ax.set_axisbelow(True)
    ax.scatter(x1, y1, c=t2, vmin=5, vmax=21, edgecolors="k", s=100)
    ax.scatter(
        [x1_filament_epoch1_mean],
        [y1_filament_epoch1_mean],
        c=storm1color,
        marker="*",
        s=500,
        edgecolors="k",
        linewidths=2,
    )
    ax.scatter(
        [x1_core_epoch1_mean],
        [y1_core_epoch1_mean],
        c=epoch1color,
        marker="*",
        s=500,
        edgecolors="k",
        linewidths=2,
    )
    ax.scatter(
        [x1_core_epoch2_mean],
        [y1_core_epoch2_mean],
        c=epoch2color,
        marker="*",
        s=500,
        edgecolors="k",
        linewidths=2,
    )
    xa = x1_core_epoch1_mean
    ya = y1_core_epoch1_mean
    dx = x1_core_epoch2_mean - x1_core_epoch1_mean
    dy = 0
    xa += 0.3 * dx
    ya += 0.1 * dy
    dx *= 0.6
    dy *= 0.4
    ax.arrow(
        xa,
        ya,
        dx,
        dy,
        width=0.04,
        length_includes_head=True,
        head_width=0.15,
        head_length=0.007,
        fill=True,
        facecolor="grey",
        edgecolor="k",
    )
    ax.set_xlabel("Sea Surface Salinity")
    ax.set_ylabel("Sea Surface Temp. ($\circ$C)")

    colors = [storm1color, epoch1color, epoch2color]
    cruise_periods = ["Filament (Epoch 1)", "Core (Epoch 1)", "Core (Epoch 2)"]
    handles = [Patch(color=colors[i], label=c) for i, c in enumerate(cruise_periods)]
    ax.legend(handles=handles, loc="upper right")
    mark_text(ax, "b)", 0.05, 0.95)
    fig.tight_layout()

    fig.savefig(output_filename, bbox_inches="tight", dpi=dpi, transparent=True)


if __name__ == "__main__":
    main()
