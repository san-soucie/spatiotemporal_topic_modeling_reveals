import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os
import math
import scipy
import altair as alt
from math import floor
alt.data_transformers.disable_max_rows()



from scipy.spatial.distance import braycurtis, jensenshannon
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy

PROJECT_DIR = os.getcwd()
DATA_FOLDER= os.path.join(PROJECT_DIR, 'data')
FIG_FOLDER=os.path.join(PROJECT_DIR, 'figures_eps')
FIG_FOLDER_JPG=os.path.join(PROJECT_DIR, 'figures_jpg')

FONTSIZE=12

def normalize(df):
    return df.div(df.sum(axis=1), axis=0)

def set_font_size(ax, size=FONTSIZE):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def set_cbar_font_size(cbar, size=FONTSIZE):
    set_font_size(cbar.ax, size=size)

plt.rcParams.update({'font.size': 8})

epoch1color = '#a8bbe3'
epoch2color = '#ffe186'
epoch3color = '#badba6'
storm1color = '#f6c29d'
storm2color = '#bfbebc'

topic_colors = [plt.cm.Dark2(i) for i in range(10)]

page_width = 7.48031
page_height = 9.05512
dpi = 600


data = pd.read_json('/home/sansoucie/projects/spatiotemporal_topic_modeling_reveals/data/model/rost_topic_prob.json').pivot(index='sample_time', columns='category', values='prob')
metadata = pd.read_json('/home/sansoucie/projects/spatiotemporal_topic_modeling_reveals/data/processed/meta_data.json')
x = metadata.day_of_month
y = data.to_numpy().T

colors = [plt.cm.Accent(i) for i in range(10)]
labels=[f'Community {i}' for i in range(1, 5)]
fig, (tax, ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5], 'hspace': 0}, figsize=(page_width, page_height / 2))

########
# TIMELINE

delta = (metadata.cruise_day - metadata.day_of_month).iloc[0]
storm1start = 2 - delta
storm1end = 6 - delta
storm2start = 8.5-delta
storm2end = 11.2-delta

tax.axis('off')
tax.set_xlim([min(x), max(x)])
tax.set_ylim([0, 1])
epoch1 = matplotlib.patches.Rectangle((min(x), 0.05), 11 - min(x)-0.025, 0.95, linewidth=2, edgecolor='k', facecolor=epoch1color)
tax.add_patch(epoch1)
tax.text(min(x) + (11 - min(x)-0.025) / 2, 0.65, 'Epoch 1', horizontalalignment='center')
epoch2 = matplotlib.patches.Rectangle((11+0.025, 0.05), max(x) - (11)-0.05, 0.95, linewidth=2, edgecolor='k', facecolor=epoch2color)
tax.add_patch(epoch2)
tax.text(11+0.025 + (max(x) - (11)-0.05) / 2, 0.65, 'Epoch 2', horizontalalignment='center')
storm1 = matplotlib.patches.Rectangle((storm1start, 0.15), storm1end-storm1start, 0.35, linewidth=2, edgecolor='k', facecolor=storm1color)
tax.add_patch(storm1)
tax.text(storm1start + (storm1end-storm1start) / 2, 0.25, 'Storm 1', horizontalalignment='center')
storm2 = matplotlib.patches.Rectangle((storm2start, 0.15), storm2end-storm2start, 0.35, linewidth=2, edgecolor='k', facecolor=storm2color)
tax.add_patch(storm2)
tax.text(storm2start + (storm2end-storm2start) / 2, 0.25, 'Storm 2', horizontalalignment='center')

########

c = ax.stackplot(x, y, colors=topic_colors, labels=labels)
legend = ax.legend()
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 0.6))

ax.set_xlim([min(x), max(x)])
ax.set_ylim([0, 1])

ax.set_xlabel('Day of month (May 2021)')
ax.set_ylabel('Community proportion')

#set_font_size(ax)

plt.show()
