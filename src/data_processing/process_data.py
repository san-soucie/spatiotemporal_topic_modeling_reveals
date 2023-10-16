import pandas as pd
from pathlib import Path

project_dir = Path(__file__).parents[2]


def main():
    raw_data_path = project_dir / "data" / "raw"
    processed_data_path = project_dir / "data" / "processed"

    classcount_path = raw_data_path / "otz_classcount.csv"
    meta_data_path = raw_data_path / "otz_meta_data.csv"
    sdgcorr_path = raw_data_path / "SdGcorr.csv"
    classlist_path = raw_data_path / "IFCB_classlist_type.csv"
    auxiliary_meta_data_path = raw_data_path / "auxiliary_meta_data.json"

    classcount_filename = processed_data_path / "classcount.json"
    meta_data_filename = processed_data_path / "meta_data.json"
    data_filename = processed_data_path / "data.json"
    rost_input_observations_filename = processed_data_path / "observations.csv"

    classcount = pd.read_csv(classcount_path, usecols=lambda x: x != "pid")
    meta_data = pd.read_csv(meta_data_path, parse_dates=["datetime"])
    auxiliary_meta_data = pd.read_json(auxiliary_meta_data_path)
    SdGcorr = pd.read_csv(sdgcorr_path, parse_dates=["datetime"])

    classes = pd.read_csv(classlist_path, index_col="CNN_classlist")
    classes = classes.fillna(0)
    classes = classes.astype(int)

    meta_data.index = meta_data.datetime
    classcount.index = meta_data.datetime

    SdGcorr.index = SdGcorr.datetime
    SdGcorr = SdGcorr.reindex(SdGcorr.index.union(meta_data.index))
    SdGcorr = SdGcorr.drop("datetime", axis=1)
    SdGcorr = SdGcorr.interpolate(method="time")
    SdGcorr = SdGcorr.reindex(meta_data.index)

    idx = ~meta_data.latitude.isna()
    idx &= ~meta_data.skip
    idx &= meta_data.sample_type == "underway"
    idx &= ~SdGcorr.latec.isna()

    classcount = classcount[idx]

    classcount.to_json(classcount_filename, orient="records")

    meta_data = meta_data[idx]
    SdGcorr = SdGcorr[idx]

    drop_cols = ["cast", "niskin", "tag1", "tag2", "tag3", "comment_summary"]
    meta_data = meta_data.drop(drop_cols, axis=1)

    intermediate_meta_data = pd.concat([meta_data, SdGcorr], axis=1)

    bad_classes_idx = classes["OtherNotAlive"] == 1
    bad_classes_idx |= classes["IFCBArtifact"] == 1
    bad_classes_idx |= classes["Detritus"] == 1
    bad_classes = classes[bad_classes_idx].index.to_list()
    bad_classes += ["pollen", "detritus_transparent", "nanoplankton_mix"]
    good_classes_data = classcount[
        [c for c in classcount.columns if c not in bad_classes]
    ]
    badmeta = ["lon", "lat", "depth"]

    intermediate_meta_data = intermediate_meta_data[
        [c for c in intermediate_meta_data.columns if c not in badmeta]
    ]
    good_data_idx = intermediate_meta_data.sample_type == "underway"
    good_data_idx &= ~intermediate_meta_data.longitude.isna()
    good_data_idx &= ~intermediate_meta_data.skip

    good_data = good_classes_data[good_data_idx]
    processed_meta_data = intermediate_meta_data[good_data_idx]
    processed_meta_data = processed_meta_data.drop(
        ["sample_type", "skip", "dataset"], axis=1
    )
    processed_meta_data["sample_time"] = processed_meta_data.datetime.apply(
        pd.to_datetime
    ).apply(lambda t: t.tz_localize(None))
    good_data.index = processed_meta_data.sample_time

    auxiliary_meta_data.sample_time = auxiliary_meta_data.sample_time.apply(
        pd.to_datetime
    )
    auxiliary_meta_data = auxiliary_meta_data.set_index("sample_time")
    processed_meta_data = processed_meta_data.set_index("sample_time")
    for c in [x for x in auxiliary_meta_data.columns if x != "ml_analyzed"]:
        processed_meta_data[c] = auxiliary_meta_data[c]
    processed_meta_data = processed_meta_data.dropna()

    good_data = good_data[good_data.index.isin(processed_meta_data.index)]

    processed_meta_data = processed_meta_data.reset_index()
    processed_meta_data.to_json(meta_data_filename, orient="records")
    good_data.to_json(data_filename, orient="records")

    observations = good_data.reset_index()
    observations = observations.drop(["sample_time"], axis=1)
    observations.to_csv(rost_input_observations_filename, index=False)


if __name__ == "__main__":
    main()
