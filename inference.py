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
EVAL_DIR = Path("training/exp2")


def main():
    device = "cuda"
    hp_path = "config/hparams.yaml"
    # cp_path = "aver_checkpoints/aver_g-194_196_220_240_244_246_252_257_265_271_276_283_194_196_220_244_257_265_271_276_283"
    # cp_path = "checkpoints/g_00000320"
    cp_path = "checkpoints/g_00000329"

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
