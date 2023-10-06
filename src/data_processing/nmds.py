import pandas as pd
import numpy as np
from ..utils import DATA_DIR, get_config

from scipy.spatial.distance import braycurtis
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA


def nmds():
    config = get_config()
    model_dir = DATA_DIR / config["data"]["model_subdir_name"]
    nmds_filename = model_dir / 'nmds.json'
    wt_matrix_filename = model_dir / 'rost_wt_matrix.json'
    word_prob_filename = model_dir / 'rost_word_prob.json'

    random_seed = config["random_seed"]

    observations = pd.read_json(filelist.data)
    metadata = pd.read_json(filelist.meta_data)

    obs_dissimilarity = pairwise_distances(observations.to_numpy(), metric=braycurtis)

    nmds = MDS(n_init=1, n_components=2, metric=False, dissimilarity='precomputed', random_state=RANDOM_SEED)
    nmds.fit(obs_dissimilarity, init=PCA(n_components=2).fit_transform(obs_dissimilarity))
    x = nmds.embedding_[..., 0]
    y = nmds.embedding_[..., 1]
    t = metadata.day_of_month.to_numpy()
    stress = np.sqrt(nmds.stress_ / (0.5 * np.sum(obs_dissimilarity ** 2)))

    df = pd.DataFrame({'nmds_component_1': x, 'nmds_component_2': y, 'day_of_month': t, 'stress': [stress] * len(x)})
    df.to_json(filelist.nmds, orient='records')
    return filelist