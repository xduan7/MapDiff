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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from dataloader.large_dataset import Cath
from dataloader.collator import CollatorDiff
from model.egnn_pytorch.egnn_net import EGNN_NET
from model.ipa.ipa_net import IPANetPredictor
from model.prior_diff import Prior_Diff
from evaluator import Evaluator
from utils import enable_dropout, set_seed
from prettytable import PrettyTable
from datetime import datetime
import time
import torch.cuda.amp as amp


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


class EnsembleDataset(Dataset):
    """Dataset that creates (protein, ensemble_index) pairs for distributed processing."""
    
    def __init__(self, base_dataset, ensemble_num):
        self.base_dataset = base_dataset
        self.ensemble_num = ensemble_num
        self.collator = CollatorDiff()
    
    def __len__(self):
        return len(self.base_dataset) * self.ensemble_num
    
    def __getitem__(self, idx):
        protein_idx = idx // self.ensemble_num
        ensemble_idx = idx % self.ensemble_num
        protein_data = self.base_dataset[protein_idx]
        return protein_data, protein_idx, ensemble_idx


def evaluate_dataset_single_gpu(model, dataloader, evaluator, device, cfg, dataset_name="Dataset"):
    """Original single-GPU evaluation function."""
    return evaluate_dataset(model, dataloader, evaluator, device, cfg, dataset_name)


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
    
    # Get optimization settings
    use_amp = hasattr(cfg.evaluation, 'use_amp') and cfg.evaluation.use_amp and device.type == 'cuda'
    fast_eval = hasattr(cfg.evaluation, 'fast_eval') and cfg.evaluation.fast_eval
    ensemble_num = cfg.evaluation.fast_ensemble_num if fast_eval else cfg.evaluation.ensemble_num
    
    if use_amp:
        print(f"Using automatic mixed precision (AMP)")
    if fast_eval:
        print(f"Fast evaluation mode: ensemble size reduced to {ensemble_num}")
    
    with torch.no_grad():
        for batch_idx, (g_batch, ipa_batch) in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}", 
                                      disable=not cfg.logging.verbose)):
            g_batch = g_batch.to(device)
            ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
            
            # Ensemble predictions
            ens_logits = []
            
            if use_amp:
                # Use automatic mixed precision for faster inference
                with amp.autocast():
                    for _ in range(ensemble_num):
                        logits, _ = model.mc_ddim_sample(g_batch, ipa_batch, 
                                                       diverse=True, 
                                                       step=cfg.evaluation.ddim_steps)
                        # Detach and move to CPU immediately to free GPU memory
                        ens_logits.append(logits.detach().cpu())  
            else:
                for _ in range(ensemble_num):
                    logits, _ = model.mc_ddim_sample(g_batch, ipa_batch, 
                                                   diverse=True, 
                                                   step=cfg.evaluation.ddim_steps)
                    # Detach and move to CPU immediately to free GPU memory
                    ens_logits.append(logits.detach().cpu())
            
            # Stack and compute mean on CPU to save GPU memory
            ens_logits = torch.stack(ens_logits)
            mean_logits = ens_logits.mean(dim=0)  # Already on CPU
            # Free ensemble logits memory
            del ens_logits
            
            all_logits.append(mean_logits)
            all_seq.append(g_batch.x.detach().cpu())
            
            # Calculate per-sample metrics (each batch has only one sample due to batch_size=1)
            sample_logits = mean_logits.argmax(dim=1)  # Shape: [sequence_length]
            sample_seq = all_seq[-1].argmax(dim=1)  # Use already detached CPU tensor
            sample_recovery = evaluator.cal_recovery(sample_logits, sample_seq)
            
            if evaluator.blosum_eval:
                sample_nssr42, sample_nssr62, sample_nssr80, sample_nssr90 = evaluator.cal_all_blosum_nssr(
                    sample_logits, sample_seq)
                nssr42.append(sample_nssr42)
                nssr62.append(sample_nssr62)
                nssr80.append(sample_nssr80)
                nssr90.append(sample_nssr90)
            
            recovery.append(sample_recovery)
            
            # Clear GPU cache periodically based on batch index
            if device.type == 'cuda' and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
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
    
    # Check if multi-GPU is requested and available
    use_multi_gpu = hasattr(cfg.evaluation, 'use_multi_gpu') and cfg.evaluation.use_multi_gpu
    num_gpus = torch.cuda.device_count() if use_multi_gpu else 1
    
    if use_multi_gpu and num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel evaluation")
        # Multi-GPU evaluation will be handled differently
        mp.set_start_method('spawn', force=True)
    else:
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
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Initialize model only for single GPU
    if not (use_multi_gpu and num_gpus > 1):
        print("\nInitializing models...")
        model, device = initialize_model(cfg)
    
    # Evaluate each dataset
    all_results = {}
    collator = CollatorDiff()
    
    for dataset_name, dataset in datasets_to_eval.items():
        if use_multi_gpu and num_gpus > 1:
            # Multi-GPU evaluation
            start_time = time.time()
            results = evaluate_dataset_multi_gpu(
                dataset, cfg, dataset_name.capitalize() + " Set", num_gpus
            )
            elapsed_time = time.time() - start_time
        else:
            # Single GPU evaluation
            dataloader = DataLoader(
                dataset, 
                batch_size=cfg.evaluation.batch_size, 
                shuffle=False,
                num_workers=cfg.evaluation.num_workers, 
                collate_fn=collator
            )
            
            # Initialize models for single GPU
            if 'model' not in locals():
                model, device = initialize_model(cfg)
            
            start_time = time.time()
            results = evaluate_dataset(model, dataloader, evaluator, device, cfg, 
                                     dataset_name.capitalize() + " Set")
            elapsed_time = time.time() - start_time
        results['evaluation_time_seconds'] = elapsed_time
        results['evaluation_time_minutes'] = elapsed_time / 60
        all_results[dataset_name] = results
        
        # Create and print table
        table = create_results_table(results, dataset_name.capitalize() + " Set")
        print(f"\n{dataset_name.capitalize()} Set Results:")
        print(table)
        print(f"Evaluation time: {elapsed_time/60:.2f} minutes")
        
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


def initialize_model(cfg, device=None):
    """Initialize and load the model."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
    print(f"Loading checkpoint from: {cfg.evaluation.checkpoint_path}")
    checkpoint = torch.load(cfg.evaluation.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'step' in checkpoint:
        print(f"Checkpoint from step: {checkpoint['step']}")
    
    # Optimization: Compile model if PyTorch 2.0+ and enabled
    use_compile = hasattr(cfg.evaluation, 'use_compile') and cfg.evaluation.use_compile
    if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile for faster inference...")
        model = torch.compile(model, mode='reduce-overhead')
    
    return model, device


def worker_fn(rank, world_size, cfg, dataset, dataset_name, result_queue):
    """Worker function for multi-GPU evaluation."""
    # Set up distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.evaluation.get('master_port', '12355'))
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    set_seed()
    
    # Initialize model on this GPU
    model, _ = initialize_model(cfg, device)
    model.eval()
    enable_dropout(model)
    
    # Create ensemble dataset
    ensemble_dataset = EnsembleDataset(dataset, cfg.evaluation.ensemble_num)
    sampler = DistributedSampler(
        ensemble_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    # Custom collate function
    def ensemble_collate_fn(batch):
        proteins = [item[0] for item in batch]
        protein_indices = [item[1] for item in batch]
        ensemble_indices = [item[2] for item in batch]
        
        collator = CollatorDiff()
        g_batch, ipa_batch = collator(proteins)
        
        return g_batch, ipa_batch, protein_indices, ensemble_indices
    
    dataloader = DataLoader(
        ensemble_dataset,
        batch_size=cfg.evaluation.get('batch_size_per_gpu', 1),
        sampler=sampler,
        num_workers=cfg.evaluation.num_workers // world_size,
        collate_fn=ensemble_collate_fn,
        pin_memory=True
    )
    
    # Evaluation loop
    local_results = []
    use_amp = cfg.evaluation.use_amp and device.type == 'cuda'
    
    with torch.inference_mode():
        for batch_idx, (g_batch, ipa_batch, protein_indices, ensemble_indices) in enumerate(
            tqdm(dataloader, desc=f"GPU {rank}", disable=rank != 0)
        ):
            g_batch = g_batch.to(device)
            ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = model.mc_ddim_sample(
                        g_batch, ipa_batch, 
                        diverse=True, 
                        step=cfg.evaluation.ddim_steps
                    )
            else:
                logits, _ = model.mc_ddim_sample(
                    g_batch, ipa_batch, 
                    diverse=True, 
                    step=cfg.evaluation.ddim_steps
                )
            
            # Store results
            for i in range(len(protein_indices)):
                local_results.append({
                    'protein_idx': protein_indices[i],
                    'ensemble_idx': ensemble_indices[i],
                    'logits': logits[i].detach().cpu(),
                    'seq': g_batch.x[i].detach().cpu()
                })
            
            # Clear cache periodically
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
    # Gather results to rank 0
    gathered_results = [None] * world_size if rank == 0 else None
    dist.gather_object(local_results, gathered_results, dst=0)
    
    if rank == 0:
        result_queue.put(gathered_results)
    
    dist.destroy_process_group()


def evaluate_dataset_multi_gpu(dataset, cfg, dataset_name, num_gpus):
    """Multi-GPU evaluation using distributed ensemble-data parallelism."""
    world_size = min(num_gpus, cfg.evaluation.get('max_gpus', num_gpus))
    
    # Use multiprocessing queue to get results
    result_queue = mp.Queue()
    
    # Spawn worker processes
    mp.spawn(
        worker_fn,
        args=(world_size, cfg, dataset, dataset_name, result_queue),
        nprocs=world_size,
        join=True
    )
    
    # Get results from queue
    gathered_results = result_queue.get()
    
    # Process results
    all_results = []
    for gpu_results in gathered_results:
        all_results.extend(gpu_results)
    
    # Organize by protein and aggregate ensemble predictions
    protein_results = {}
    for result in all_results:
        protein_idx = result['protein_idx']
        if protein_idx not in protein_results:
            protein_results[protein_idx] = {
                'logits': [],
                'seq': result['seq']
            }
        protein_results[protein_idx]['logits'].append(result['logits'])
    
    # Calculate metrics
    evaluator = Evaluator()
    recovery = []
    all_logits = []
    all_seq = []
    
    for protein_idx in sorted(protein_results.keys()):
        # Stack ensemble predictions and compute mean
        ensemble_logits = torch.stack(protein_results[protein_idx]['logits'])
        mean_logits = ensemble_logits.mean(dim=0)
        seq = protein_results[protein_idx]['seq']
        
        all_logits.append(mean_logits)
        all_seq.append(seq)
        
        # Calculate recovery
        pred = mean_logits.argmax(dim=1)
        true_seq = seq.argmax(dim=1)
        sample_recovery = evaluator.cal_recovery(pred, true_seq)
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
        'num_samples': len(recovery),
        'num_gpus_used': world_size
    }
    
    # Add BLOSUM metrics if available
    if evaluator.blosum_eval:
        nssr42, nssr62, nssr80, nssr90 = [], [], [], []
        for protein_idx in sorted(protein_results.keys()):
            ensemble_logits = torch.stack(protein_results[protein_idx]['logits'])
            mean_logits = ensemble_logits.mean(dim=0)
            seq = protein_results[protein_idx]['seq']
            
            pred = mean_logits.argmax(dim=1)
            true_seq = seq.argmax(dim=1)
            
            sample_nssr42, sample_nssr62, sample_nssr80, sample_nssr90 = evaluator.cal_all_blosum_nssr(
                pred, true_seq)
            nssr42.append(sample_nssr42)
            nssr62.append(sample_nssr62)
            nssr80.append(sample_nssr80)
            nssr90.append(sample_nssr90)
        
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


if __name__ == "__main__":
    main()