import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from infer_stuff import test_procedure, validation_epoch_end
from src.dataset import AudioFeatDataset
from src.eval_testset import _calc_embed, _cut_lines_with_dur, _load_data_from_dir
from src.model import Model
from src.utils import load_hparams, read_lines, write_lines

LOGGER = logging.getLogger(__name__)
EVAL_DIR = Path("training/exp8")

{
    "training/exp2/": {
        "checkpoints/g_00000359": {
            "rranks": 0.5635675196647386,
            "mAP": 0.2712094629849707,
        },
        "checkpoints/g_00000381": {
            "rranks": 0.5639672545611649,
            "mAP": 0.2707566079560851,
        },
        "aver_checkpoints/aver_g-295_298_300_304_313_322_329_334_337_347_355_363_368_298_304_332_337_347_356_363_368_380": {
            "rranks": 0.5712707204977174,
            "mAP": 0.2760716027649133,
        },
    },
    "training/exp3/": {
        "checkpoints/g_00000844": {
            "rranks": 0.5943551346718862,
            "mAP": 0.28666194649999815,
        },
        "aver_checkpoints/aver_g-508_536_554_599_617_643_667_681_745_766_832_845_508_554_588_599_617_681_745_832_845": {
            "rranks": 0.5970237744671043,
            "mAP": 0.2977531806381228,
        },
    },
    "training/exp4/": {
        "checkpoints/g_00000147": {
            "rranks": 0.5857654604067418,
            "mAP": 0.2932351803487232,
        },
        "aver_checkpoints/aver_g-20_25_31_20_25_28": {
            "rranks": 0.6284825086127235,
            "mAP": 0.31904239974687604,
        },
    },
    "training/exp5/": {
        "checkpoints/g_00000068": {
            "rranks": 0.6572507548901262,
            "mAP": 0.3452827979296185,
        },
        "aver_checkpoints/aver_g-53_59_65_67_68_59_62_65_68": {
            "rranks": 0.6609223754462198,
            "mAP": 0.34687882992287294,
        },
    },
    "training/exp6/": {
        "aver_checkpoints/aver_g-56_65_77_65_77_94": {
            "cosin": {
                "rranks": 0.6690447018471737,
                "mAP": 0.35702033009676626,
            },
            "euclidean": {
                "rranks": 0.6688983273321215,
                "mAP": 0.3593995205196027,
            },
            "l2": {
                "rranks": 0.6688983273321215,
                "mAP": 0.3593995205196027,
            },
            "cityblock": {
                "rranks": 0.6620381325851775,
                "mAP": 0.3462932329056158,
            },
            "l1": {
                "rranks": 0.6620381325851775,
                "mAP": 0.3462932329056158,
            },
            "manhattan": {
                "rranks": 0.6620381325851775,
                "mAP": 0.3462932329056158,
            },
        },
    },
    "training/exp7/": {
        "checkpoints/g_00000001": {
            "rranks": 0.6695665384617365,
            "mAP": 0.36923216605733294,
        },
    },
    "training/exp8/": {
        "checkpoints/g_00000237": {
            "rranks": 0.7186598306985081,
            "mAP": 0.4021911709342964,
        },
        "checkpoints/g_00000238": {
            "rranks": 0.7186891710143936,
            "mAP": 0.40239440002585897,
        },
        "aver_checkpoints/aver_g-164_182_201_219_237_164_183_201_219_237": {
            "rranks": 0.7159825631849236,
            "mAP": 0.39992498028895757,
        },
    },
}


def main():
    device = "cuda"
    hp_path = "config/hparams.yaml"
    # cp_path = "aver_checkpoints/aver_g-20_25_31_20_25_28"
    cp_path = "checkpoints/g_00000238"

    version2clique = None
    # version2clique = load_version2clique()

    # query_path = "/mnt/yandex_cup24/CoverHunterMPS_data/covers80/val.txt"
    query_path = "/mnt/yandex_cup24/CoverHunterMPS_data/ya_testset/test.txt"
    # query_path = "/mnt/yandex_cup24/CoverHunterMPS_data/covers80/query_ref.txt"

    # embed_dir = "test_eval_by_averaged_model"
    # embed_dir = "val_eval"
    embed_dir = "test_eval"

    hp = load_hparams(str(EVAL_DIR / hp_path))
    model = Model(hp).to(device)
    state_dict = torch.load(str(EVAL_DIR / cp_path), map_location=device)
    model.load_state_dict(state_dict["generator"])

    evaluate(hp, model, str(EVAL_DIR / embed_dir), query_path, version2clique)


def load_version2clique():
    data_path = "/home/sazerlife/projects/contests/yandex_cup24/dataset/"
    cliques_subset = np.load(
        os.path.join(data_path, "splits", "{}_cliques.npy".format("val"))
    )
    versions = pd.read_csv(
        os.path.join(data_path, "cliques2versions.tsv"),
        sep="\t",
        converters={"versions": eval},
    )
    versions = versions[versions["clique"].isin(set(cliques_subset))]
    mapping = {}
    for k, clique in enumerate(sorted(cliques_subset)):
        mapping[clique] = k
    versions["clique"] = versions["clique"].map(lambda x: mapping[x])
    versions.set_index("clique", inplace=True)
    version2clique = pd.DataFrame(
        [
            {"version": version, "clique": clique}
            for clique, row in versions.iterrows()
            for version in row["versions"]
        ]
    ).set_index("version")
    return version2clique


def evaluate(
    hp,
    model: Model,
    embed_dir: str,
    query_path: str,
    # ref_path: str,
    version2clique: pd.DataFrame = None,
    batch_size: int = 256,
    num_workers: int = 1,
    device: str = "cuda",
):
    os.makedirs(embed_dir, exist_ok=True)
    model.eval()
    model = model.to(device)

    infer_frame = 50  # hardcoded from eval_testset.py and hp
    chunk_s = 60

    query_lines = read_lines(query_path, log=False)

    # === Queries creating ===
    query_embed_dir = os.path.join(embed_dir, "query_embed")
    query_chunked_lines = _cut_lines_with_dur(query_lines, chunk_s, query_embed_dir)
    write_lines(os.path.join(embed_dir, "query.txt"), query_chunked_lines, False)
    data_loader = DataLoader(
        AudioFeatDataset(
            hp, data_lines=query_chunked_lines, mode="defined", chunk_len=infer_frame
        ),
        num_workers=num_workers,
        shuffle=False,
        sampler=None,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=None,
        drop_last=False,
    )
    LOGGER.info("Start to calc_embed for query")
    _calc_embed(model, data_loader, device, saved_dir=query_embed_dir)

    _, query_embed = _load_data_from_dir(query_chunked_lines)

    if version2clique is not None:
        result = validation_epoch_end(query_embed, version2clique)
        print(result)
        print("rranks", result["rranks"].mean())
        print("mAP", result["average_precisions"].mean())
    test_procedure(query_embed, embed_dir, device)

    # LOGGER.info("Start to generate dist mstricMPS")
    # dist_matrix, query_label, ref_label = _generate_dist_matrixMPS(
    #     query_perf_label,
    #     query_embed,
    #     ref_perf_label,
    #     ref_embed,
    #     query_in_ref=query_in_ref,
    # )
    # print(dist_matrix)
    # metrics = calc_map(dist_matrix, query_label, ref_label, topk=10000, verbose=0)
    # print(metrics)


if __name__ == "__main__":
    main()
