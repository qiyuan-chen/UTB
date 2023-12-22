from ast import arg
import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Any
import argparse

class CheckpointManager:
    def __init__(self, model: Module, optimizer: Optimizer, args: argparse.Namespace):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = args.checkpoint_dir

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(state, f"{self.checkpoint_dir}/checkpoint_{epoch}.pth")

    def load_checkpoint(self, checkpoint_path: str) -> Optional[int]:
        try:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            return checkpoint.get('epoch', None)
        except FileNotFoundError:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
