from detectron2.utils.events import get_event_storage, EventStorage, EventWriter
from detectron2.config import CfgNode

from typing import Union, Dict


class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(
        self,
        project: str = "detectron2",
        config: Union[Dict, CfgNode] = {},  # noqa: B006
        window_size: int = 20,
        **kwargs,
    ):
        """
        Args:
            project (str): W&B Project name
            config Union[Dict, CfgNode]: the project level configuration object
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `wandb.init(...)`
        """
        import wandb

        self._window_size = window_size
        self._run = (
            wandb.init(project=project, config=config, **kwargs)
            if not wandb.run
            else wandb.run
        )
        self._run._label(repo="detectron2")

    def write(self):
        storage = get_event_storage()

        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            self._run.log({k: v}, step=iter)

    def close(self):
        self._run.finish()
