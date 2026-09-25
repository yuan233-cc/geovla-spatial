from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import IterableDataset

from prismatic.training.strategies import base_strategy
from prismatic.training.strategies.base_strategy import TrainingStrategy


class _Dataset(IterableDataset):
    def __iter__(self):
        for scale in (1.0, 3.0):
            yield {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "pixel_values": None,
                "labels": torch.tensor([[0, 1, 2]]),
                "dataset_names": [b"libero_spatial"],
                "scale": scale,
            }

    def __len__(self):
        return 2


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.vision_backbone = SimpleNamespace(num_patches=0)

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        del input_ids, attention_mask, pixel_values
        # The batch-specific scalar is encoded in the last label for this test.
        scale = labels[0, -1].float()
        loss = self.weight * scale
        logits = torch.zeros((1, labels.shape[1], 4))
        return SimpleNamespace(loss=loss, logits=logits)


class _Tokenizer:
    action_token_begin_idx = -1

    @staticmethod
    def decode_token_ids_to_actions(token_ids):
        return np.asarray(token_ids, dtype=np.float32)


class _Metrics:
    def __init__(self):
        self.global_step = 0
        self.run_dir = Path(".")

    @staticmethod
    def get_status():
        return "test"

    def commit(self, global_step=None, **kwargs):
        del kwargs
        if global_step is not None:
            self.global_step = global_step

    @staticmethod
    def commit_for_dataset(**kwargs):
        del kwargs

    @staticmethod
    def push():
        return "test"


class _Strategy(TrainingStrategy):
    def __init__(self, model):
        self.vlm = model
        self.per_device_batch_size = 1
        self.global_batch_size = 2
        self.grad_accumulation_steps = 2
        self.epochs = 1
        self.max_steps = 1
        self.enable_mixed_precision_training = False
        self.mixed_precision_dtype = torch.bfloat16
        self.worker_init_fn = None
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.saved_steps = []
        self.clip_calls = 0

    def save_checkpoint(self, run_dir, global_step, epoch, train_loss=None, only_trainable=True):
        del run_dir, epoch, train_loss, only_trainable
        self.saved_steps.append(global_step)

    def run_setup(self, run_dir, n_train_examples):
        del run_dir, n_train_examples

    def clip_grad_norm(self):
        self.clip_calls += 1


def test_vla_gradient_accumulation_steps_once_and_averages_gradients():
    model = _Model()
    strategy = _Strategy(model)
    metrics = _Metrics()

    # Use the sample scalar as the model's final label. The two micro-batches
    # therefore contribute gradients 1 and 3; their accumulated mean is 2.
    dataset = _Dataset()

    def collate(rows):
        batch = rows[0]
        batch["labels"][0, -1] = int(batch.pop("scale"))
        return batch

    with (
        mock.patch.object(base_strategy.overwatch, "world_size", return_value=1),
        mock.patch.object(base_strategy.overwatch, "is_rank_zero", return_value=True),
        mock.patch.object(torch.distributed, "barrier"),
    ):
        strategy.run_vla_training(dataset, collate, _Tokenizer(), metrics)

    assert torch.allclose(model.weight.detach(), torch.tensor(0.8))
    assert metrics.global_step == 1
    assert strategy.clip_calls == 1
    assert strategy.saved_steps == [1]
