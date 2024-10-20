from pathlib import Path
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import jsonlines

tqdm.pandas()

DATASET_DIR = Path("/home/sazerlife/projects/contests/yandex_cup24/dataset")
# COVERS_DIR = Path("/home/sazerlife/projects/contests/yandex_cup24/CoverHunterMPS_my/data")
COVERS_DIR = Path("/mnt/yandex_cup24/CoverHunterMPS_data/")


def main():
    cliques2versions = pd.read_csv(DATASET_DIR / "cliques2versions.tsv", sep="\t")
    train_cliques_id = np.load(str(DATASET_DIR / "splits/train_cliques.npy"))
    valid_cliques_id = np.load(str(DATASET_DIR / "splits/val_cliques.npy"))
    # train = cliques2versions.loc[cliques2versions["clique"].isin(train_cliques_id)]
    # valid = cliques2versions.loc[cliques2versions["clique"].isin(valid_cliques_id)]
    # full_processing(valid, COVERS_DIR / "covers80_testset")
    full_processing(
        cliques2versions, COVERS_DIR / "covers80", train_cliques_id, valid_cliques_id
    )


def full_processing(
    df: pd.DataFrame,
    work_dir: Path,
    train_cliques_id: np.ndarray,
    valid_cliques_id: np.ndarray,
):
    dataset = crate_dataset_txt(df, work_dir)
    work_id_map = crate_work_id_map(df, work_dir)
    full_txt = create_full_txt(dataset, work_id_map, work_dir)

    valid_txt = full_txt.loc[
        full_txt["work"].apply(lambda x: int(x)).isin(valid_cliques_id)
    ]
    query, ref = create_ref_query(valid_txt)
    query.to_json(work_dir / "query.txt", orient="records", lines=True)
    ref.to_json(work_dir / "ref.txt", orient="records", lines=True)

    train_txt = full_txt.loc[
        full_txt["work"].apply(lambda x: int(x)).isin(train_cliques_id)
    ]
    train_txt.to_json(work_dir / "train.txt", orient="records", lines=True)


def crate_dataset_txt(df: pd.DataFrame, work_dir: Path):
    columns = ["perf", "wav", "dur_s", "work", "version"]
    dataset_txt = list()
    wav_path_plug = str(work_dir / "wav_16k/null.wav")

    pbar = tqdm(df.values, desc=f"{work_dir.name}: dataset.txt / data.init.txt")
    for clique, versions in pbar:
        versions = ast.literal_eval(versions)
        for version in versions:
            line = (str(version), wav_path_plug, 60.0, str(clique), str(version))
            dataset_txt.append(line)

    dataset_txt = pd.DataFrame(dataset_txt, columns=columns)
    dataset_txt.to_json(work_dir / "data.init.txt", orient="records", lines=True)
    dataset_txt.to_json(work_dir / "dataset.txt", orient="records", lines=True)
    return dataset_txt


def crate_work_id_map(df: pd.DataFrame, work_dir: Path):
    id2clique = dict()
    clique2id = dict()
    pbar = tqdm(
        df.sort_values(by="clique").values, desc=f"{work_dir.name}: work_id.map"
    )
    for id_, (clique, _) in enumerate(pbar):
        id2clique[id_] = clique
        clique2id[clique] = id_

    pd.DataFrame(list(id2clique.items())).to_csv(
        work_dir / "work_id.map", sep=" ", index=None, header=None
    )
    return clique2id


def create_full_txt(dataset_txt: pd.DataFrame, work_id_map: dict, work_dir: Path):
    full_txt = dataset_txt.copy()
    feat, feat_len, work_id = list(), list(), list()
    feat_len_plug = 50
    pbar = tqdm(full_txt[["version", "work"]].values, desc=f"{work_dir.name}: full.txt")
    for version, work in pbar:
        feat.append(str(work_dir / f"cqt_feat/{version}.npy"))
        feat_len.append(feat_len_plug)
        work_id.append(work_id_map[int(work)])

    full_txt["feat"], full_txt["feat_len"] = feat, feat_len
    full_txt["work_id"] = work_id
    print(full_txt)
    full_txt.to_json(work_dir / "full.txt", orient="records", lines=True)
    return full_txt


def create_ref_query(valid_txt: pd.DataFrame):
    query = list()
    ref = list()
    pbar = tqdm(valid_txt.groupby("work"), desc=f"ref.txt / query.txt")
    for work_name, subdf in pbar:
        query.append(subdf.iloc[1].values)
        ref.append(subdf.iloc[0].values)

    return (
        pd.DataFrame(query, columns=valid_txt.columns),
        pd.DataFrame(ref, columns=valid_txt.columns),
    )


if __name__ == "__main__":
    main()
