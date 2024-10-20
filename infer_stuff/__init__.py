import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import pairwise_distances_chunked, pairwise_distances
from tqdm import tqdm
import pandas as pd


def test_procedure(
    predicted_embed: Dict[str, List[np.ndarray]], output_dir: str, device: str = "cuda"
) -> None:
    trackids: List[int] = []
    embeddings: List[np.array] = []
    print(len(predicted_embed.items()))
    for anchor_id, sub_embeddings in predicted_embed.items():
        assert len(sub_embeddings) == 1, f"Too many embeddings: {sub_embeddings}"
        trackids.append(anchor_id)
        embeddings.append(sub_embeddings[0])
    print(len(embeddings))

    predictions = []
    for chunk_result in pairwise_distances_chunked(
        embeddings, metric="cosine", reduce_func=reduce_func, working_memory=1000
    ):
        for query_indx, query_nearest_items in chunk_result:
            predictions.append(
                (
                    trackids[query_indx],
                    [trackids[nn_indx] for nn_indx in query_nearest_items],
                )
            )
    save_test_predictions(predictions, output_dir)


def reduce_func(D_chunk, start):
    top_size = 100
    nearest_items = np.argsort(D_chunk, axis=1)[:, : top_size + 1]
    return [(i, items[items != i]) for i, items in enumerate(nearest_items, start)]


def save_test_predictions(predictions: List, output_dir: str) -> None:
    with open(os.path.join(output_dir, "submission.txt"), "w") as foutput:
        for query_item, query_nearest in predictions:
            foutput.write(
                "{}\t{}\n".format(query_item, "\t".join(map(str, query_nearest)))
            )


def validation_epoch_end(
    outputs: Dict[str, List[np.ndarray]], version2clique: pd.DataFrame
) -> Dict[int, np.ndarray]:
    clique_ids = []
    embeddings: List[np.array] = []
    for anchor_id, sub_embeddings in outputs.items():
        assert len(sub_embeddings) == 1, f"Too many embeddings: {sub_embeddings}"
        clique_ids.append(version2clique.loc[int(anchor_id), "clique"])
        embeddings.append(sub_embeddings[0])

    # preds = torch.stack(list(outputs.values()))[:, 1]
    preds = np.asarray(embeddings)
    rranks, average_precisions = calculate_ranking_metrics(
        embeddings=preds, cliques=clique_ids
    )
    return {
        # "triplet_ids": np.stack(list(zip(clique_ids, anchor_ids, pos_ids, neg_ids))),
        "rranks": rranks,
        "average_precisions": average_precisions,
    }


def calculate_ranking_metrics(
    embeddings: np.ndarray, cliques: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(embeddings, metric="cosine")
    s_distances = np.argsort(distances, axis=1)
    cliques = np.array(cliques)
    query_cliques = cliques[s_distances[:, 0]]
    search_cliques = cliques[s_distances[:, 1:]]

    query_cliques = np.tile(query_cliques, (search_cliques.shape[-1], 1)).T
    mask = np.equal(search_cliques, query_cliques)

    ranks = 1.0 / (mask.argmax(axis=1) + 1.0)

    cumsum = np.cumsum(mask, axis=1)
    mask2 = mask * cumsum
    mask2 = mask2 / np.arange(1, mask2.shape[-1] + 1)
    average_precisions = np.sum(mask2, axis=1) / np.sum(mask, axis=1)

    return (ranks, average_precisions)
