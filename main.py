from comet_ml import Experiment
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import Trainer
from dataloader.large_dataset import Cath
from model.egnn_pytorch.egnn_net import EGNN_NET
from model.ipa.ipa_net import IPANetPredictor
from torch.utils.data import DataLoader
from model.prior_diff import Prior_Diff
from torch.optim import Adam, lr_scheduler
from utils import set_seed
from dataloader.collator import CollatorDiff


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="conf", config_name="diff_config")
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
    set_seed()

    if cfg.dataset.name == 'CATH':
        train_ID, val_ID, test_ID = os.listdir(cfg.dataset.train_dir), os.listdir(cfg.dataset.val_dir), \
            os.listdir(cfg.dataset.test_dir)
        train_dataset = Cath(train_ID, cfg.dataset.train_dir)
        val_dataset = Cath(val_ID, cfg.dataset.val_dir)
        test_dataset = Cath(test_ID, cfg.dataset.test_dir)
        print(f'Train on CATH dataset with {len(train_dataset)} training data, {len(val_dataset)} '
              f'val data, {len(test_dataset)}  test data')
    else:
        raise ValueError(f"unknown dataset")

    collator = CollatorDiff()

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8,
                                  collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=8,
                                collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=8,
                                 collate_fn=collator)

    train_num_steps = len(train_dataloader) * cfg.train.train_epochs + 1

    model = EGNN_NET(input_feat_dim=cfg.model.input_feat_dim, hidden_channels=cfg.model.hidden_dim,
                     edge_attr_dim=cfg.model.edge_attr_dim,
                     dropout=cfg.model.drop_out, n_layers=cfg.model.depth, update_edge=cfg.model.update_edge,
                     norm_coors=cfg.model.norm_coors, update_coors=cfg.model.update_coors,
                     update_global=cfg.model.update_global, embedding=cfg.model.embedding,
                     embedding_dim=cfg.model.embedding_dim, norm_feat=cfg.model.norm_feat, embed_ss=cfg.model.embed_ss)

    prior_model = IPANetPredictor(dropout=cfg.model.ipa_drop_out)
    prior_checkpoint = torch.load(cfg.prior_model.path)
    prior_model.load_state_dict(prior_checkpoint['model'], strict=False)

    diffusion_model = Prior_Diff(model, prior_model, timesteps=cfg.diffusion.timesteps,
                                 objective=cfg.diffusion.objective,
                                 noise_type=cfg.diffusion.noise_type, sample_method=cfg.diffusion.sample_method,
                                 min_mask_ratio=cfg.mask_prior.min_mask_ratio,
                                 dev_mask_ratio=cfg.mask_prior.dev_mask_ratio,
                                 marginal_dist_path=cfg.dataset.marginal_train_dir,
                                 ensemble_num=cfg.diffusion.ensemble_num)

    print(f"Total parameters: {count_parameters(diffusion_model)}")

    optimizer = Adam(diffusion_model.parameters(), lr=cfg.train.lr, betas=(0.95, 0.999),
                     weight_decay=cfg.train.weight_decay)
    if cfg.train.scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=train_num_steps)
    else:
        scheduler = None

    trainer = Trainer(cfg,
                      diffusion_model,
                      train_dataloader,
                      val_dataloader,
                      test_dataloader,
                      optimizer,
                      device,
                      output_dir,
                      scheduler=scheduler,
                      train_num_steps=train_num_steps,
                      save_and_sample_every=cfg.train.save_and_sample_every,
                      train_batch_size=cfg.train.batch_size,
                      ddim_steps=cfg.diffusion.ddim_steps,
                      sample_method=cfg.diffusion.sample_method,
                      ensemble_num=cfg.diffusion.ensemble_num,
                      experiment=experiment)
    trainer.train()
    trainer.test()
    trainer.save_table_results()


if __name__ == "__main__":
    main()
