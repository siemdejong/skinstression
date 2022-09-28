"""
The BaseAgent class. New agents must inherit from BaseAgent, as BaseAgent serves as a template for new agents.
"""
import logging


class BaseAgent:
    """This base class will contain the base functions to be implemented by any new agent."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, filename: str):
        """Load latest checkpoint.

        Args:
            filename: name of checkpoint file in directory provided by config.checkpoint_dir.
        """
        raise NotImplementedError

    def save_checkpoint(
        self, filename: str = "checkpoint.pth.tar", is_best: bool = False
    ):
        """Saves checkpoint.

        Args:
            filename: name of checkpoint file in directory provide by config.checkpoint_dir to be loaded.
            is_best: indicate whether current checkpoint's metric is the best so far.
        """
        raise NotImplementedError

    def run(self):
        """Main operator."""
        raise NotImplementedError

    def train(self):
        """Main training loop."""
        raise NotImplementedError

    def train_one_epoch(self):
        """One epoch of training."""
        raise NotImplementedError

    def validate(self):
        """One cycle of model validation."""
        raise NotImplementedError

    def finalize(self):
        """Finalize all operations of this agent and corresponding dataloader"""
        raise NotImplementedError
