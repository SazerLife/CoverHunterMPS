from pathlib import Path
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import jsonlines

tqdm.pandas()

{
    "perf": "200383",
    "wav": "\/mnt\/yandex_cup24\/CoverHunterMPS_data\/covers80\/wav_16k\/null.wav",
    "dur_s": 60.0,
    "work": "10003",
    "version": "200383",
    "feat": "\/mnt\/yandex_cup24\/CoverHunterMPS_data\/covers80\/cqt_feat\/200383.npy",
    "feat_len": 50,
    "work_id": 10002,
}


DATASET_DIR = Path("/home/sazerlife/projects/contests/yandex_cup24/dataset")
COVERS_DIR = Path("/mnt/yandex_cup24/CoverHunterMPS_data/")


def main():
    work_dir = COVERS_DIR / "ya_testset"
    dataset_txt = create_testset(work_dir)
    dataset_txt.to_json(work_dir / "test.txt", orient="records", lines=True)


def create_testset(work_dir: Path):
    npy_paths = list((work_dir / "cqt_feat").rglob("*.npy"))

    columns = ["perf", "wav", "dur_s", "work", "version", "feat", "feat_len", "work_id"]
    dataset_txt = list()
    wav_path_plug = str(work_dir / "wav_16k/null.wav")
    feat_len_plug = 50
    for npy_path in tqdm(npy_paths):
        line = (
            str(npy_path.stem),
            wav_path_plug,
            60.0,
            "-1",
            str(npy_path.stem),
            str(npy_path),
            feat_len_plug,
            -1,
        )
        dataset_txt.append(line)
    return pd.DataFrame(dataset_txt, columns=columns)


if __name__ == "__main__":
    main()
