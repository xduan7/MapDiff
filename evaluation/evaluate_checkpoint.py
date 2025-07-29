from comet_ml import Experiment
import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from dataloader.large_dataset import Cath
from dataloader.collator import CollatorDiff
from model.egnn_pytorch.egnn_net import EGNN_NET
from model.ipa.ipa_net import IPANetPredictor
from model.prior_diff import Prior_Diff
from evaluator import Evaluator
from utils import enable_dropout, set_seed
from prettytable import PrettyTable
from datetime import datetime


def cal_stats_metric(values):
    mean_value = np.mean(values)
    median_value = np.median(values)
    std_value = np.std(values)
    return mean_value, median_value, std_value


def save_results_json(results, filepath):
    """Save results as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)


def save_results_csv(results, filepath):
    """Save results as CSV file."""
    # Flatten nested dictionaries
    rows = []
    for dataset_name, metrics in results.items():
        for metric_name, value in metrics.items():
            rows.append({
                'dataset': dataset_name,
                'metric': metric_name,
                'value': value
            })
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(model, dataloader, evaluator, device, cfg, dataset_name="Dataset"):
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    enable_dropout(model)
    
    all_logits = []
    all_seq = []
    recovery = []
    nssr42, nssr62, nssr80, nssr90 = [], [], [], []
    
    if cfg.logging.verbose:
        print(f"\nEvaluating on {dataset_name}...")
        print(f"Total batches: {len(dataloader)}")
    
    with torch.no_grad():
        for g_batch, ipa_batch in tqdm(dataloader, desc=f"Processing {dataset_name}", 
                                      disable=not cfg.logging.verbose):
            g_batch = g_batch.to(device)
            ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
            
            # Ensemble predictions
            ens_logits = []
            for _ in range(cfg.evaluation.ensemble_num):
                logits, _ = model.mc_ddim_sample(g_batch, ipa_batch, 
                                               diverse=True, 
                                               step=cfg.evaluation.ddim_steps)
                ens_logits.append(logits)
            
            ens_logits = torch.stack(ens_logits)
            mean_logits = ens_logits.mean(dim=0).cpu()  # Shape: [sequence_length, num_classes]
            
            all_logits.append(mean_logits)
            all_seq.append(g_batch.x.cpu())
            
            # Calculate per-sample metrics (each batch has only one sample due to batch_size=1)
            sample_logits = mean_logits.argmax(dim=1)  # Shape: [sequence_length]
            sample_seq = g_batch.x.cpu().argmax(dim=1)  # Shape: [sequence_length]
            sample_recovery = evaluator.cal_recovery(sample_logits, sample_seq)
            
            if evaluator.blosum_eval:
                sample_nssr42, sample_nssr62, sample_nssr80, sample_nssr90 = evaluator.cal_all_blosum_nssr(
                    sample_logits, sample_seq)
                nssr42.append(sample_nssr42)
                nssr62.append(sample_nssr62)
                nssr80.append(sample_nssr80)
                nssr90.append(sample_nssr90)
            
            recovery.append(sample_recovery)
    
    # Aggregate metrics
    all_logits = torch.cat(all_logits)
    all_seq = torch.cat(all_seq)
    
    mean_recovery, median_recovery, std_recovery = cal_stats_metric(recovery)
    full_recovery = (all_logits.argmax(dim=1) == all_seq.argmax(dim=1)).sum() / all_seq.shape[0]
    full_recovery = full_recovery.item()
    perplexity = evaluator.cal_perplexity(all_logits, all_seq)
    
    results = {
        'mean_recovery': mean_recovery,
        'median_recovery': median_recovery,
        'std_recovery': std_recovery,
        'full_recovery': full_recovery,
        'perplexity': perplexity,
        'num_samples': len(recovery)
    }
    
    if evaluator.blosum_eval:
        mean_nssr42, median_nssr42, std_nssr42 = cal_stats_metric(nssr42)
        mean_nssr62, median_nssr62, std_nssr62 = cal_stats_metric(nssr62)
        mean_nssr80, median_nssr80, std_nssr80 = cal_stats_metric(nssr80)
        mean_nssr90, median_nssr90, std_nssr90 = cal_stats_metric(nssr90)
        
        results.update({
            'mean_nssr42': mean_nssr42, 'median_nssr42': median_nssr42, 'std_nssr42': std_nssr42,
            'mean_nssr62': mean_nssr62, 'median_nssr62': median_nssr62, 'std_nssr62': std_nssr62,
            'mean_nssr80': mean_nssr80, 'median_nssr80': median_nssr80, 'std_nssr80': std_nssr80,
            'mean_nssr90': mean_nssr90, 'median_nssr90': median_nssr90, 'std_nssr90': std_nssr90
        })
    
    return results


def create_results_table(results, dataset_name):
    """Create a PrettyTable from results dictionary."""
    table = PrettyTable()
    table.field_names = ["Metric", dataset_name]
    
    # Main metrics
    table.add_row(["Median Recovery", f"{results['median_recovery']:.4f}"])
    table.add_row(["Mean Recovery", f"{results['mean_recovery']:.4f}"])
    table.add_row(["Std Recovery", f"{results['std_recovery']:.4f}"])
    table.add_row(["Full Recovery", f"{results['full_recovery']:.4f}"])
    table.add_row(["Perplexity", f"{results['perplexity']:.4f}"])
    table.add_row(["Num Samples", results['num_samples']])
    
    # BLOSUM metrics if available
    if 'median_nssr42' in results:
        table.add_row(["---", "---"])
        for blosum in [42, 62, 80, 90]:
            table.add_row([f"Median NSSR{blosum}", f"{results[f'median_nssr{blosum}']:.4f}"])
            table.add_row([f"Mean NSSR{blosum}", f"{results[f'mean_nssr{blosum}']:.4f}"])
    
    return table


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate_checkpoint")
def main(cfg: DictConfig):
    # Initialize Comet ML if requested
    experiment = None
    if cfg.logging.use_comet and cfg.comet.use:
        experiment = Experiment(
            project_name=cfg.comet.project_name,
            workspace=cfg.comet.workspace,
            auto_output_logging="simple",
            log_graph=False,
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
        experiment.add_tag("evaluation")
    
    # Get output directory from Hydra
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    print("="*80)
    print(f"MapDiff Checkpoint Evaluation")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {cfg.evaluation.checkpoint_path}")
    print("="*80)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed()
    
    # Load datasets
    print("\nLoading datasets...")
    datasets_to_eval = {}
    
    if cfg.evaluation.evaluate_val:
        val_ID = os.listdir(cfg.dataset.val_dir)
        val_dataset = Cath(val_ID, cfg.dataset.val_dir)
        datasets_to_eval['validation'] = val_dataset
        print(f"Validation dataset size: {len(val_dataset)}")
    
    if cfg.evaluation.evaluate_test:
        test_ID = os.listdir(cfg.dataset.test_dir)
        test_dataset = Cath(test_ID, cfg.dataset.test_dir)
        datasets_to_eval['test'] = test_dataset
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize models
    print("\nInitializing models...")
    egnn = EGNN_NET(
        input_feat_dim=cfg.model.input_feat_dim,
        hidden_channels=cfg.model.hidden_dim,
        edge_attr_dim=cfg.model.edge_attr_dim,
        dropout=cfg.model.drop_out,
        n_layers=cfg.model.depth,
        update_edge=cfg.model.update_edge,
        norm_coors=cfg.model.norm_coors,
        update_coors=cfg.model.update_coors,
        update_global=cfg.model.update_global,
        embedding=cfg.model.embedding,
        embedding_dim=cfg.model.embedding_dim,
        norm_feat=cfg.model.norm_feat,
        embed_ss=cfg.model.embed_ss
    )
    
    ipa = IPANetPredictor(dropout=cfg.model.ipa_drop_out)
    
    model = Prior_Diff(
        egnn, ipa,
        timesteps=cfg.diffusion.timesteps,
        objective=cfg.diffusion.objective,
        noise_type=cfg.diffusion.noise_type,
        sample_method=cfg.diffusion.sample_method,
        min_mask_ratio=cfg.mask_prior.min_mask_ratio,
        dev_mask_ratio=cfg.mask_prior.dev_mask_ratio,
        marginal_dist_path=cfg.dataset.marginal_train_dir,
        ensemble_num=cfg.evaluation.ensemble_num
    ).to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {cfg.evaluation.checkpoint_path}")
    checkpoint = torch.load(cfg.evaluation.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'step' in checkpoint:
        print(f"Checkpoint from step: {checkpoint['step']}")
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Evaluate each dataset
    all_results = {}
    collator = CollatorDiff()
    
    for dataset_name, dataset in datasets_to_eval.items():
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.evaluation.batch_size, 
            shuffle=False,
            num_workers=cfg.evaluation.num_workers, 
            collate_fn=collator
        )
        
        results = evaluate_dataset(model, dataloader, evaluator, device, cfg, 
                                 dataset_name.capitalize() + " Set")
        all_results[dataset_name] = results
        
        # Create and print table
        table = create_results_table(results, dataset_name.capitalize() + " Set")
        print(f"\n{dataset_name.capitalize()} Set Results:")
        print(table)
        
        # Log to Comet ML if enabled
        if experiment:
            for metric, value in results.items():
                experiment.log_metric(f"{dataset_name}_{metric}", value)
    
    # Save results in various formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cfg.logging.save_tables:
        # Save pretty tables
        tables_path = os.path.join(output_dir, "evaluation_tables.txt")
        with open(tables_path, "w") as f:
            f.write(f"MapDiff Evaluation Results\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Checkpoint: {cfg.evaluation.checkpoint_path}\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, results in all_results.items():
                table = create_results_table(results, dataset_name.capitalize() + " Set")
                f.write(f"{dataset_name.capitalize()} Set Results:\n")
                f.write(table.get_string() + "\n\n")
        
        print(f"\nTables saved to: {tables_path}")
    
    if cfg.logging.save_json:
        # Save as JSON
        json_path = os.path.join(output_dir, "evaluation_results.json")
        json_data = {
            'timestamp': timestamp,
            'checkpoint': cfg.evaluation.checkpoint_path,
            'config': OmegaConf.to_container(cfg),
            'results': all_results
        }
        save_results_json(json_data, json_path)
        print(f"JSON results saved to: {json_path}")
    
    if cfg.logging.save_csv:
        # Save as CSV
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        save_results_csv(all_results, csv_path)
        print(f"CSV results saved to: {csv_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"MapDiff Evaluation Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {cfg.evaluation.checkpoint_path}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("\nKey Results:\n")
        
        for dataset_name, results in all_results.items():
            f.write(f"\n{dataset_name.capitalize()} Set:\n")
            f.write(f"  - Median Recovery: {results['median_recovery']:.4f}\n")
            f.write(f"  - Perplexity: {results['perplexity']:.4f}\n")
            if 'median_nssr62' in results:
                f.write(f"  - Median NSSR62: {results['median_nssr62']:.4f}\n")
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nAll results saved to: {output_dir}")
    
    if experiment:
        experiment.end()


if __name__ == "__main__":
    main()