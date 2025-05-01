import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import seed_everything


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg):
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.seed)
    datamodule = instantiate(cfg.dataset)
    # NOTE: _recursive_=False for avoid instantiation of loss modules
    model = instantiate(cfg.model, _recursive_=False)
    if "pretrained_ckpt" in cfg and cfg.pretrained_ckpt:
        state_dict = torch.load(cfg.pretrained_ckpt)["state_dict"]
        model.load_state_dict(state_dict)
    # NOTE: _convert_="partial" for neptune tags (list instead of ListConfig)
    trainer = instantiate(cfg.trainer, _convert_="partial")
    trainer.logger.log_hyperparams(cfg)
    if "resume_ckpt" in cfg and cfg.resume_ckpt:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.resume_ckpt)
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
