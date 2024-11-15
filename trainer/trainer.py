import torch
import torch.nn.functional as F
import os
import copy
import numpy as np
import datetime
from tqdm import tqdm
from utils import inf_iterator, enable_dropout, cal_stats_metric
from pathlib import Path
from omegaconf import OmegaConf
from prettytable import PrettyTable
from evaluator import Evaluator


class Trainer(object):
    def __init__(
            self,
            config,
            diffusion_model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            optimizer,
            device,
            output_dir,
            scheduler=None,
            train_batch_size=512,
            train_num_steps=200000,
            save_and_sample_every=100,
            num_samples=25,
            ensemble_num=50,
            ddim_steps=50,
            sample_method='ddim',
            experiment=None
    ):
        super().__init__()
        self.device = device
        self.model = diffusion_model.to(self.device)
        self.config = config
        self.num_samples = num_samples
        self.ensemble_num = ensemble_num
        self.ddim_steps = ddim_steps
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps
        self.sample_method = sample_method

        # dataset and dataloader
        self.train_dataloader = train_dataloader
        self.iter_one_epoch = len(train_dataloader)
        self.train_iterator = inf_iterator(train_dataloader)
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        # optimizer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = Evaluator()
        self.best_val_step = 0
        self.best_val_epoch = 0
        self.step = 0
        self.epoch = 0
        self.best_val_recovery, self.best_val_perplexity = 0, float('inf')
        self.best_model = None

        self.train_metric_header = ["# Epoch", "# Step", "Train_loss"]
        self.val_metric_header = ["# Epoch", "# Step", "Recovery", "Perplexity"]
        self.test_metric_header = ["# Epoch", "# Step", "Recovery", "Perplexity"]
        self.train_table = PrettyTable(self.train_metric_header)
        self.val_table = PrettyTable(self.val_metric_header)
        self.test_table = PrettyTable(self.test_metric_header)

        self.results_folder = output_dir
        Path(self.results_folder + '/model/').mkdir(exist_ok=True)
        self.experiment = experiment

    def save(self, save_epochs, save_steps, mode='best'):
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        if mode == 'best':
            data = {
                'config': config_dict,
                'step': save_steps,
                'epoch': save_epochs,
                'model': self.best_model.state_dict(),
                'opt': self.optimizer.state_dict(),
            }
        elif mode == 'last':
            data = {
                'config': config_dict,
                'step': save_steps,
                'epoch': save_epochs,
                'model': self.model.state_dict(),
                'opt': self.optimizer.state_dict(),
            }
        else:
            raise ValueError(f"unknown mode {mode}")
        save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(data, os.path.join(self.results_folder, 'model',
                                      f'{self.config.experiment.name}_{mode}_{save_epochs}_epochs_{save_steps}_steps_{save_time}.pt'))

    def save_table_results(self):
        with open(os.path.join(self.results_folder, 'train_markdowntable.txt'), 'w') as f:
            f.write(self.train_table.get_string())
        with open(os.path.join(self.results_folder, 'val_markdowntable.txt'), 'w') as f:
            f.write(self.val_table.get_string())
        with open(os.path.join(self.results_folder, 'test_markdowntable.txt'), 'w') as f:
            f.write(self.test_table.get_string())

    def train(self):
        epoch_total_loss = 0
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()
                g_batch, ipa_batch = next(self.train_iterator)
                g_batch, ipa_batch = g_batch.to(self.device), ipa_batch.to(self.device) if ipa_batch is not None else None
                base_loss, mask_loss = self.model(g_batch, ipa_batch)
                loss = base_loss + mask_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1
                epoch_total_loss += loss.item()

                if self.experiment:
                    self.experiment.log_metric('train_base_loss', base_loss.item(), step=self.step, epoch=self.epoch)
                    self.experiment.log_metric('train_mask_loss', mask_loss.item(), step=self.step, epoch=self.epoch)
                    self.experiment.log_metric('train_loss', loss.item(), step=self.step, epoch=self.epoch)

                if self.step % self.iter_one_epoch == 0 and self.step != 0:
                    self.epoch += 1
                    self.train_table.add_row([self.epoch, self.step, epoch_total_loss / self.iter_one_epoch])
                    epoch_total_loss = 0
                    torch.cuda.empty_cache()

                if self.step != 0 and self.step % (self.save_and_sample_every * self.iter_one_epoch) == 0:
                    self.model.eval()
                    enable_dropout(self.model)
                    with torch.no_grad():
                        all_logits = torch.tensor([])
                        all_seq = torch.tensor([])
                        recovery = []
                        for g_batch, ipa_batch in tqdm(self.val_dataloader):
                            g_batch, ipa_batch = g_batch.to(self.device), ipa_batch.to(self.device) if ipa_batch is not None else None
                            ens_logits = []
                            if self.sample_method == 'ddim':
                                for _ in range(self.ensemble_num):
                                    logits, sample_graph = self.model.mc_ddim_sample(g_batch, ipa_batch, diverse=True,
                                                                                  step=self.ddim_steps)
                                    ens_logits.append(logits)
                            ens_logits_tensor = torch.stack(ens_logits)
                            batch_logits = ens_logits_tensor.mean(dim=0).cpu()
                            all_logits = torch.cat([all_logits, batch_logits])
                            all_seq = torch.cat([all_seq, g_batch.x.cpu()])

                            # given all_logits and g_batch.batch, calculate recovery rate for each sequence
                            batch_idx = g_batch.batch.cpu().numpy()
                            for i in range(batch_idx.max() + 1):
                                idx = np.where(batch_idx == i)
                                sample_logits = batch_logits[idx].argmax(dim=1)
                                sample_seq = g_batch.x.cpu()[idx].argmax(dim=1)
                                sample_recovery = self.evaluator.cal_recovery(sample_logits, sample_seq)
                                recovery.append(sample_recovery)

                        mean_recovery, median_recovery = cal_stats_metric(recovery)

                        full_recovery = (all_logits.argmax(dim=1) == all_seq.argmax(dim=1)).sum() / all_seq.shape[0]
                        full_recovery = full_recovery.item()

                        perplexity = self.evaluator.cal_perplexity(all_logits, all_seq)
                        print()
                        print(f'Val median recovery rate (step: {self.step}) is {median_recovery}')
                        print(f'Val perplexity (step: {self.step}): {perplexity}')
                        self.val_table.add_row([self.epoch, self.step, median_recovery, perplexity])
                        if self.experiment:
                            self.experiment.log_metric('val_full_recovery', full_recovery, epoch=self.epoch)
                            self.experiment.log_metric('val_perplexity', perplexity, epoch=self.epoch)
                            self.experiment.log_metric('val_median_recovery', median_recovery, epoch=self.epoch)
                            self.experiment.log_metric('val_mean_recovery', mean_recovery, epoch=self.epoch)

                        if median_recovery > self.best_val_recovery:
                            self.best_model = copy.deepcopy(self.model)
                            self.best_val_step = self.step
                            self.best_val_epoch = self.epoch
                            self.best_val_recovery = median_recovery
                            self.best_val_perplexity = perplexity
                pbar.update(1)
        print('Training complete')
        if self.experiment:
            self.experiment.log_metric('best_val_median_recovery', self.best_val_recovery)
            self.experiment.log_metric('best_val_perplexity', self.best_val_perplexity)
            self.experiment.log_metric('best_val_epoch', self.best_val_epoch)
        self.save(self.best_val_epoch, self.best_val_step, mode='best')
        self.save(self.epoch, self.train_num_steps, mode='last')

    def test(self):
        self.best_model.eval()
        enable_dropout(self.best_model)
        with torch.no_grad():
            print('Testing best model')
            all_logits = torch.tensor([])
            all_seq = torch.tensor([])
            recovery = []
            nssr42, nssr62, nssr80, nssr90 = [], [], [], []
            for g_batch, ipa_batch in self.test_dataloader:
                g_batch, ipa_batch = g_batch.to(self.device), ipa_batch.to(self.device) if ipa_batch is not None else None
                ens_logits = []
                if self.sample_method == 'ddim':
                    for _ in range(self.ensemble_num):
                        logits, sample_graph = self.best_model.mc_ddim_sample(g_batch, ipa_batch, diverse=True, step=self.ddim_steps)
                        ens_logits.append(logits)
                ens_logits_tensor = torch.stack(ens_logits)
                batch_logits = ens_logits_tensor.mean(dim=0).cpu()
                all_logits = torch.cat([all_logits, batch_logits])
                all_seq = torch.cat([all_seq, g_batch.x.cpu()])

                batch_idx = g_batch.batch.cpu().numpy()
                for i in range(batch_idx.max() + 1):
                    idx = np.where(batch_idx == i)
                    sample_logits = batch_logits[idx].argmax(dim=1)
                    sample_seq = g_batch.x.cpu()[idx].argmax(dim=1)
                    sample_recovery = self.evaluator.cal_recovery(sample_logits, sample_seq)
                    sample_nssr42, sample_nssr62, sample_nssr80, sample_nssr90 = self.evaluator.cal_all_blosum_nssr(sample_logits, sample_seq)
                    nssr42.append(sample_nssr42)
                    nssr62.append(sample_nssr62)
                    nssr80.append(sample_nssr80)
                    nssr90.append(sample_nssr90)
                    recovery.append(sample_recovery)

            test_mean_recovery, test_median_recovery = cal_stats_metric(recovery)
            test_mean_nssr42, test_median_nssr42 = cal_stats_metric(nssr42)
            test_mean_nssr62, test_median_nssr62 = cal_stats_metric(nssr62)
            test_mean_nssr80, test_median_nssr80 = cal_stats_metric(nssr80)
            test_mean_nssr90, test_median_nssr90 = cal_stats_metric(nssr90)

            test_recovery = (all_logits.argmax(dim=1) == all_seq.argmax(dim=1)).sum() / all_seq.shape[0]
            test_recovery = test_recovery.item()
            test_perplexity = self.evaluator.cal_perplexity(all_logits, all_seq)
            print(f'test median recovery rate with best model (step: {self.best_val_step}) is {test_median_recovery}')
            print(f'test perplexity with the best model (step: {self.best_val_step}) is: {test_perplexity}')
            self.test_table.add_row([self.best_val_epoch, self.best_val_step, test_median_recovery, test_perplexity])
            if self.experiment:
                self.experiment.log_metric('test_full_recovery_with_best_model', test_recovery)
                self.experiment.log_metric('test_perplexity_with_best_model', test_perplexity)
                self.experiment.log_metric('test_median_recovery_with_best_model', test_median_recovery)
                self.experiment.log_metric('test_mean_recovery_with_best_model', test_mean_recovery)
                self.experiment.log_metric('test_median_nssr42_with_best_model', test_median_nssr42)
                self.experiment.log_metric('test_median_nssr62_with_best_model', test_median_nssr62)
                self.experiment.log_metric('test_median_nssr80_with_best_model', test_median_nssr80)
                self.experiment.log_metric('test_median_nssr90_with_best_model', test_median_nssr90)
