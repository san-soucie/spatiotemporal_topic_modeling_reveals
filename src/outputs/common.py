from pathlib import Path

project_dir = Path(__file__).parents[2]

epoch1color = "#a8bbe3"
epoch2color = "#ffe186"
epoch3color = "#badba6"
storm1color = "#f6c29d"
storm2color = "#bfbebc"


def mark_text(ax, text, dx, dy):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    x = minx + dx * (maxx - minx)
    y = miny + dy * (maxy - miny)
    ax.text(x, y, text, backgroundcolor="white")
