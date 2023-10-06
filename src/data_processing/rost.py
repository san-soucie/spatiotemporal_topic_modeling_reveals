import rostpy

import yaml

import pandas as pd
from ..utils import ConfigDict, FileList, ROSTConfig
from pathlib import Path
from click import pass_context, Context
import click
from ..__main__ import cli, data_options


@cli.command()
@data_options
@click.option('--force-overwrite', is_flag=True, default=False) # cannot be specified except from command line
@pass_context
def rost(ctx: Context, force_overwrite: bool = False, **kwargs) -> FileList:
    config: ConfigDict = ctx.obj[ConfigDict.name]
    params: ROSTConfig = config.__getattribute__(ROSTConfig.name)
    filelist = ctx.obj[FileList.name]

    filelist.topic_prob = config.model_data_path / 'rost_topic_prob.json'
    filelist.wt_matrix = config.model_data_path / 'rost_wt_matrix.json'
    filelist.word_prob = config.model_data_path / 'rost_word_prob.json'

    ctx.obj[FileList.name] = filelist

    if not kwargs['dry_run'] and force_overwrite:
        dataset = pd.read_csv(filelist.rost_input_observations, header=0).fillna(0).astype(int)
        taxa = dataset.columns
        meta_data = pd.read_json(filelist.meta_data)
        times = meta_data.sample_time
        rost = rostpy.ROST_t(
            V=len(taxa),
            K=params.k,
            alpha=params.alpha,
            beta=params.beta,
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

        topic_prob_df_melted.to_json(filelist.topic_prob, orient="records")
        wt_matrix_df_norm_melted.to_json(filelist.wt_matrix, orient="records")
        word_prob_df_melted.to_json(filelist.word_prob, orient="records")

    return filelist