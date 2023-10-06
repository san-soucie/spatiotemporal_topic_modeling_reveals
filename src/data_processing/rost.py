import shutil

import pandas as pd
from pathlib import Path
from dvc.api import params_show

project_dir = Path(__file__).parents[2]


def rost():
    params = params_show()
    overwrite = params["rost"]["overwrite"]
    copy_preserved_data = params["rost"]["copy_preserved_data"]
    topic_prob_filename = project_dir / "data" / "model" / "rost_topic_prob.json"
    wt_matrix_filename = project_dir / "data" / "model" / "rost_wt_matrix.json"
    word_prob_filename = project_dir / "data" / "model" / "rost_word_prob.json"

    preserved_topic_prob_filename = (
        project_dir / "data" / "preserved_rost_output" / "rost_topic_prob.json"
    )
    preserved_wt_matrix_filename = (
        project_dir / "data" / "preserved_rost_output" / "rost_wt_matrix.json"
    )
    preserved_word_prob_filename = (
        project_dir / "data" / "preserved_rost_output" / "rost_word_prob.json"
    )

    rost_input_observations_filename = project_dir / "processed" / "observations.csv"
    metadata_filename = project_dir / "data" / "processed" / "data.json"

    has_rostpy = False
    try:
        import rostpy

        has_rostpy = True
    except ImportError:
        pass

    if not overwrite:
        topic_prob_filename.touch()
        word_prob_filename.touch()
        wt_matrix_filename.touch()
    elif copy_preserved_data:
        shutil.copy2(preserved_topic_prob_filename, topic_prob_filename)
        shutil.copy2(preserved_word_prob_filename, word_prob_filename)
        shutil.copy2(preserved_wt_matrix_filename, wt_matrix_filename)
    elif has_rostpy:
        dataset = (
            pd.read_csv(rost_input_observations_filename, header=0)
            .fillna(0)
            .astype(int)
        )
        taxa = dataset.columns
        meta_data = pd.read_json(metadata_filename)
        times = meta_data.sample_time
        rost = rostpy.ROST_t(
            V=len(taxa),
            K=int(params["rost"]["k"]),
            alpha=float(params["rost"]["alpha"]),
            beta=float(params["rost"]["beta"]),
            gamma=1e-5,
        )
        for t, (_, data) in enumerate(dataset.iterrows()):
            obs = []
            for taxon, count in enumerate(data):
                obs.extend([taxon] * count)
            rost.add_observation([t], obs)

        for epoch in range(params.epochs):  # epoch
            rostpy.parallel_refine(rost, 8)

        topics = {f"topic_{i+1}": [0 for _ in times] for i in range(rost.K)}
        for t in range(len(times)):
            for topic in rost.get_topics_for_pose((t,)):
                topics[f"topic_{topic+1}"][t] += 1
        topic_df = pd.DataFrame(topics, index=times)

        topic_prob_df = topic_df.div(topic_df.sum(axis=1), axis=0)
        topic_prob_df_melted = topic_prob_df.reset_index().melt(
            ["sample_time"], var_name="category", value_name="prob"
        )

        word_topic_matrix = rost.get_topic_model()
        wt_matrix_df = pd.DataFrame(
            word_topic_matrix, columns=taxa, index=[f"topic_{i}" for i in range(rost.K)]
        )
        wt_matrix_df_norm = wt_matrix_df.div(wt_matrix_df.sum(axis=1), axis=0)
        wt_matrix_df_norm_melted = (
            wt_matrix_df_norm.reset_index()
            .melt(["index"], var_name="taxon", value_name="prob")
            .rename(columns={"index": "topic"})
        )

        word_probs = topic_prob_df.to_numpy() @ wt_matrix_df_norm.to_numpy()
        word_prob_df = pd.DataFrame(word_probs, columns=taxa, index=times)
        word_prob_df_melted = word_prob_df.reset_index().melt(
            id_vars=["sample_time"], var_name="category", value_name="prob"
        )

        topic_prob_df_melted.to_json(topic_prob_filename, orient="records")
        wt_matrix_df_norm_melted.to_json(wt_matrix_filename, orient="records")
        word_prob_df_melted.to_json(word_prob_filename, orient="records")
