import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import braycurtis
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from dvc.api import params_show

project_dir = Path(__file__).parents[2]


def nmds():
    nmds_filename = project_dir / "data" / "model" / "nmds.json"
    observations_filename = project_dir / "data" / "processed" / "data.json"
    metadata_filename = project_dir / "data" / "processed" / "metadata.json"

    params = params_show()
    random_seed = params["nmds"]["random_seed"]

    observations = pd.read_json(observations_filename)
    metadata = pd.read_json(metadata_filename)

    obs_dissimilarity = pairwise_distances(observations.to_numpy(), metric=braycurtis)

    nmds = MDS(
        n_init=1,
        n_components=2,
        metric=False,
        dissimilarity="precomputed",
        random_state=random_seed,
    )
    nmds.fit(
        obs_dissimilarity, init=PCA(n_components=2).fit_transform(obs_dissimilarity)
    )
    x = nmds.embedding_[..., 0]
    y = nmds.embedding_[..., 1]
    t = metadata.day_of_month.to_numpy()
    stress = np.sqrt(nmds.stress_ / (0.5 * np.sum(obs_dissimilarity**2)))

    df = pd.DataFrame(
        {
            "nmds_component_1": x,
            "nmds_component_2": y,
            "day_of_month": t,
            "stress": [stress] * len(x),
        }
    )
    df.to_json(nmds_filename, orient="records")
