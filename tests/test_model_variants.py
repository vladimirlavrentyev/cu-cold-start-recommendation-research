import sys
from pathlib import Path
from unittest import TestCase

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "TaNP"))

from TaNP import NP
from embeddings_TaNP import Gating_Decoder


def model_config():
    return {
        "if_dim": 4,
        "uf_dim": 3,
        "first_embedding_dim": 5,
        "second_embedding_dim": 2,
        "z1_dim": 2,
        "z2_dim": 2,
        "z_dim": 2,
        "enc_h1_dim": 3,
        "enc_h2_dim": 3,
        "taskenc_h1_dim": 3,
        "taskenc_h2_dim": 3,
        "taskenc_final_dim": 2,
        "clusters_k": 2,
        "temperature": 1.0,
        "dropout_rate": 0.0,
        "dec_h1_dim": 3,
        "dec_h2_dim": 3,
        "dec_h3_dim": 3,
        "decoder": "gating_film",
    }


class ModelVariantsTest(TestCase):
    def test_selects_gating_film_decoder(self) -> None:
        model = NP(model_config())
        self.assertIsInstance(model.xz_to_y, Gating_Decoder)

    def test_compact_ids_match_dense_one_hot_embedding(self) -> None:
        model = NP(model_config())
        compact = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
        dense = torch.cat(
            (
                F.one_hot(compact[:, 0], num_classes=4),
                F.one_hot(compact[:, 1], num_classes=3),
            ),
            dim=1,
        )
        compact_embedding = model.embedding(compact)
        dense_embedding = model.embedding(dense)
        self.assertTrue(torch.equal(compact_embedding, dense_embedding))
