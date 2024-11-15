from comet_ml import Experiment
import hydra
import os
import torch
from dataloader.large_dataset import Cath
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from dataloader.collator import CollatorIPAPretrain
from model.ipa.ipa_net import IPANetPredictor
from torch.optim import Adam, lr_scheduler
from trainer.mask_ipa_trainer import Trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="conf", config_name="mask_pretrain")
def main(cfg: DictConfig):
    if cfg.comet.use:
        experiment = Experiment(
            project_name=cfg.comet.project_name,
            workspace=cfg.comet.workspace,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        experiment.log_parameters(OmegaConf.to_container(cfg))
        experiment.set_name(cfg.experiment.name)
        if cfg.comet.comet_tag:
            experiment.add_tag(cfg.comet.comet_tag)
    else:
        experiment = None
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))
    print(f"Output directory: {output_dir}")
    if experiment:
        experiment.log_parameters({"output_dir": output_dir})
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_ID = os.listdir(cfg.dataset.train_dir)

    train_dataset = Cath(train_ID, cfg.dataset.train_dir)

    collator = CollatorIPAPretrain(candi_rate=cfg.train.candi_rate, mask_rate=cfg.train.mask_rate,
                                   replace_rate=cfg.train.replace_rate, keep_rate=cfg.train.keep_rate)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=6,
                              collate_fn=collator)

    model = IPANetPredictor(dropout=cfg.model.ipa_drop_out).to(device)
    print(f"Total parameters: {count_parameters(model)}")

    steps_per_epoch = len(train_loader)
    optimizer = Adam(model.parameters(), lr=cfg.train.lr, betas=(0.95, 0.999), weight_decay=cfg.train.weight_decay)
    if cfg.train.scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=cfg.train.train_epochs * steps_per_epoch)
    else:
        scheduler = None

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    trainer = Trainer(config=cfg, model=model, optimizer=optimizer, epochs=cfg.train.train_epochs, loss_fn=loss_fn,
                      train_dataloader=train_loader, output_dir=output_dir, device=device,
                      scheduler=scheduler, experiment=experiment)

    trainer.fit()
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
