import pandas as pd

try:
    from .common import project_dir
except ImportError:
    from common import project_dir


def cruise_period_from_day(t):
    if t < 2:
        return r"Before storm 1\textsuperscript{a}"
    elif t < 6:
        return r"Storm 1\textsuperscript{b}"
    elif t < 9:
        return r"Between storms\textsuperscript{c}"
    elif t < 11:
        return r"Storm 2\textsuperscript{d}"
    else:
        return r"After storm 2\textsuperscript{e}"


CRUISE_PERIOD_SORT_VALUE = {
    r"Before storm 1\textsuperscript{a}": 1,
    r"Storm 1\textsuperscript{b}": 2,
    r"Between storms\textsuperscript{c}": 3,
    r"Storm 2\textsuperscript{d}": 4,
    r"After storm 2\textsuperscript{e}": 5,
}


def distance_class(r):
    if r < 15:
        return r"Inside\textsuperscript{f}"
    elif r < 45:
        return r"Near\textsuperscript{g}"
    else:
        return r"Far\textsuperscript{h}"


DISTANCE_CLASS_SORT_VALUE = {
    r"Inside\textsuperscript{f}": 1,
    r"Near\textsuperscript{g}": 2,
    r"Far\textsuperscript{h}": 3,
}


def main():
    data_filename = project_dir / "data" / "model" / "rost_topic_prob.json"
    metadata_filename = project_dir / "data" / "processed" / "meta_data.json"
    output_filename = project_dir / "output" / "table1.tex"

    topics = pd.read_json(data_filename).pivot(
        index="sample_time", columns="category", values="prob"
    )
    metadata = pd.read_json(metadata_filename)
    metadata["cruise_period"] = metadata.cruise_day.apply(cruise_period_from_day)
    metadata["distance_class"] = metadata.r_ec.apply(distance_class)
    topics["cruise_period"] = list(metadata.cruise_period)
    topics["distance_class"] = list(metadata.distance_class)
    df = topics.groupby(["cruise_period", "distance_class"]).apply("mean").reset_index()
    df.columns.name = ""
    df["sort1"] = df["cruise_period"].apply(lambda c: CRUISE_PERIOD_SORT_VALUE[c])
    df["sort2"] = df["distance_class"].apply(lambda c: DISTANCE_CLASS_SORT_VALUE[c])
    df.columns = [
        "Cruise period",
        "Location",
        "Com. 1",
        "Com. 2",
        "Com. 3",
        "Com. 4",
        "sort1",
        "sort2",
    ]
    df = df.sort_values(by=["sort1", "sort2"])
    df = df.drop(["sort1", "sort2"], axis=1)
    cruise_period_without_duplicates = df["Cruise period"].copy(deep=True)
    cruise_period_without_duplicates[pd.Series.duplicated(df["Cruise period"])] = ""
    df["Cruise period"] = cruise_period_without_duplicates
    with open(output_filename, "w") as f:
        f.writelines(
            df.to_latex(
                index=False, float_format="%.6f", column_format="|l|l||r|r|r|r|"
            )
        )


if __name__ == "__main__":
    main()
