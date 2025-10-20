# dam/lightning/train.py
import logging
import torch
import yaml
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from .dam_lightning_module import DamLightning
from .dam_datamodule import DamDataModule


def _to_plain(obj):
    """Recursively convert jsonargparse Namespaces and exotic Path_* objects to plain Python types."""
    basic = (str, int, float, bool, type(None))
    if isinstance(obj, basic):
        return obj
    if isinstance(obj, (list, tuple)):
        return [ _to_plain(x) for x in obj ]
    if isinstance(obj, dict):
        return { k: _to_plain(v) for k, v in obj.items() }
    try:
        d = vars(obj)
    except TypeError:
        try:
            return str(obj)
        except Exception:
            return repr(obj)
    return { k: _to_plain(v) for k, v in d.items() }


def _log_config(cli: "LightningCLI"):
    logging.info("Loaded configuration:")
    cfg_ns = getattr(cli.config, "fit", cli.config)
    cfg_dict = _to_plain(cfg_ns)
    logging.info("\n" + yaml.safe_dump(cfg_dict, sort_keys=False))

    if cli.trainer.loggers:
        for lg in cli.trainer.loggers:
            if isinstance(lg, WandbLogger):
                if getattr(cli.trainer, "overfit_batches", 0):
                    lg._log_model = False
                lg.experiment.config.update(cfg_dict, allow_val_change=True)


class MyLightningCLI(LightningCLI):
    def before_fit(self):
        _log_config(self)


def main():
    torch.set_float32_matmul_precision("medium")
    # Use subcommands like:  python -m dam.lightning.train fit --config path.yaml
    MyLightningCLI(
        model_class=DamLightning,
        datamodule_class=DamDataModule,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=333,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
