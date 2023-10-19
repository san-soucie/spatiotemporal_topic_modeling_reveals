from pathlib import Path
from matplotlib.pyplot import cm
import matplotlib

project_dir = Path(__file__).parents[2]

epoch1color = "#a8bbe3"
epoch2color = "#ffe186"
epoch3color = "#badba6"
storm1color = "#f6c29d"
storm2color = "#bfbebc"

topic_colors = [cm.Dark2(i) for i in range(10)]
taxon_colors = [
    matplotlib.cm.rainbow(((i * (17 * 23 * 29) % 80) / 80)) for i in range(40)
]


def mark_text(ax, text, dx, dy):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    x = minx + dx * (maxx - minx)
    y = miny + dy * (maxy - miny)
    ax.text(x, y, text, backgroundcolor="white")
