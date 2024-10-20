import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

# SRC_DIR = Path("/home/sazerlife/projects/contests/yandex_cup24/dataset/train")
# TRG_DIR = Path(
#     "/home/sazerlife/projects/contests/yandex_cup24/CoverHunterMPS_my/data/covers80/cqt_feat/"
# )

SRC_DIR = Path("/home/sazerlife/projects/contests/yandex_cup24/dataset/test")
TRG_DIR = Path("/mnt/yandex_cup24/CoverHunterMPS_data/ya_testset/cqt_feat")


def main():
    src_paths = list(SRC_DIR.rglob("*.npy"))
    TRG_DIR.mkdir(exist_ok=True, parents=True)
    for npy_path in tqdm(src_paths):
        cqt: np.ndarray = np.load(str(npy_path))
        np.save(str(TRG_DIR / npy_path.name), cqt.T)
        # print(cqt.transpose((1, 0)).shape)
        # break
        # shutil.copy(str(npy_path), str(TRG_DIR / npy_path.name))


if __name__ == "__main__":
    main()
