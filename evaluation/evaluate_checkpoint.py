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
# Import TorchMetrics for distributed evaluation
try:
    import torchmetrics
    from torchmetrics import Accuracy, MeanMetric
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: TorchMetrics not available. Using fallback implementation.")


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


class StreamingMetrics:
    """Streaming metrics calculator to avoid memory accumulation."""
    
    def __init__(self, device='cpu', distributed=False):
        self.device = device
        self.distributed = distributed
        self.reset()
    
    def reset(self):
        self.total_correct = torch.tensor(0, dtype=torch.long, device=self.device)
        self.total_samples = torch.tensor(0, dtype=torch.long, device=self.device)
        self.total_log_probs = torch.tensor(0.0, dtype=torch.float, device=self.device)
        self.per_sample_recoveries = []
        
        if TORCHMETRICS_AVAILABLE:
            self.accuracy = Accuracy(task="multiclass", num_classes=20).to(self.device)
            self.perplexity = MeanMetric().to(self.device)
    
    def update(self, logits, targets):
        """Update metrics with a batch of predictions."""
        preds = logits.argmax(dim=1)
        targets_idx = targets.argmax(dim=1)
        
        # Update accuracy
        correct = (preds == targets_idx).sum()
        self.total_correct += correct
        self.total_samples += targets.shape[0]
        
        # Update perplexity components
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, 1, targets_idx.unsqueeze(1)).squeeze()
        self.total_log_probs += target_log_probs.sum()
        
        # Calculate per-sample recovery
        batch_recovery = (preds == targets_idx).float().mean().item()
        self.per_sample_recoveries.append(batch_recovery)
        
        if TORCHMETRICS_AVAILABLE:
            self.accuracy.update(preds, targets_idx)
            # Calculate batch perplexity
            batch_perplexity = torch.exp(-target_log_probs.mean())
            self.perplexity.update(batch_perplexity)
    
    def compute(self):
        """Compute final metrics."""
        # Reduce metrics across processes if distributed
        if self.distributed:
            dist.all_reduce(self.total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total_log_probs, op=dist.ReduceOp.SUM)
        
        full_recovery = (self.total_correct.float() / self.total_samples.float()).item()
        perplexity = torch.exp(-self.total_log_probs / self.total_samples).item()
        
        # Calculate stats from per-sample recoveries
        mean_recovery, median_recovery, std_recovery = cal_stats_metric(self.per_sample_recoveries)
        
        results = {
            'mean_recovery': mean_recovery,
            'median_recovery': median_recovery,
            'std_recovery': std_recovery,
            'full_recovery': full_recovery,
            'perplexity': perplexity,
            'num_samples': len(self.per_sample_recoveries)
        }
        
        return results


def vectorized_ensemble_sample(model, g_batch, ipa_batch, ensemble_num, cfg, use_amp=False):
    """Use torch.vmap for efficient ensemble predictions if available."""
    # Check if torch.vmap is available
    if hasattr(torch, 'vmap') and hasattr(torch, 'func'):
        try:
            # Create a function that runs a single ensemble member
            def single_ensemble():
                return model.mc_ddim_sample(g_batch, ipa_batch, 
                                          diverse=True, 
                                          step=cfg.evaluation.ddim_steps)
            
            # Vectorize over ensemble dimension
            ensemble_fn = torch.vmap(single_ensemble, in_dims=None, out_dims=0)
            all_logits, _ = ensemble_fn()
            return all_logits
        except Exception as e:
            # Fallback to sequential if vmap fails
            print(f"torch.vmap failed, falling back to sequential: {e}")
    
    # Fallback: Sequential ensemble predictions
    ens_logits = []
    for _ in range(ensemble_num):
        if use_amp:
            with amp.autocast():
                logits, _ = model.mc_ddim_sample(g_batch, ipa_batch, 
                                               diverse=True, 
                                               step=cfg.evaluation.ddim_steps)
        else:
            logits, _ = model.mc_ddim_sample(g_batch, ipa_batch, 
                                           diverse=True, 
                                           step=cfg.evaluation.ddim_steps)
        ens_logits.append(logits.detach())
    
    return torch.stack(ens_logits)


def evaluate_dataset(model, dataloader, evaluator, device, cfg, dataset_name="Dataset"):
    """Evaluate model on a dataset with streaming metrics to avoid memory accumulation."""
    model.eval()
    enable_dropout(model)
    
    # Initialize streaming metrics
    metrics = StreamingMetrics(device=device, distributed=False)
    
    # Tracking for BLOSUM metrics
    nssr42, nssr62, nssr80, nssr90 = [], [], [], []
    
    if cfg.logging.verbose:
        print(f"\nEvaluating on {dataset_name}...")
        print(f"Total batches: {len(dataloader)}")
    
    # Get optimization settings
    use_amp = hasattr(cfg.evaluation, 'use_amp') and cfg.evaluation.use_amp and device.type == 'cuda'
    fast_eval = hasattr(cfg.evaluation, 'fast_eval') and cfg.evaluation.fast_eval
    ensemble_num = cfg.evaluation.fast_ensemble_num if fast_eval else cfg.evaluation.ensemble_num
    
    # More aggressive cache clearing frequency
    cache_clear_freq = 5 if ensemble_num > 20 else 10
    
    if use_amp:
        print(f"Using automatic mixed precision (AMP)")
    if fast_eval:
        print(f"Fast evaluation mode: ensemble size reduced to {ensemble_num}")
    
    with torch.no_grad():
        for batch_idx, (g_batch, ipa_batch) in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}", 
                                      disable=not cfg.logging.verbose)):
            g_batch = g_batch.to(device)
            ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
            
            # Get ensemble predictions
            ensemble_logits = vectorized_ensemble_sample(
                model, g_batch, ipa_batch, ensemble_num, cfg, use_amp
            )
            
            # Compute mean logits
            mean_logits = ensemble_logits.mean(dim=0)
            
            # Free ensemble memory immediately
            del ensemble_logits
            
            # Update streaming metrics
            metrics.update(mean_logits, g_batch.x)
            
            # Calculate BLOSUM metrics if needed
            if evaluator.blosum_eval:
                sample_logits = mean_logits.argmax(dim=1)
                sample_seq = g_batch.x.argmax(dim=1)
                sample_nssr42, sample_nssr62, sample_nssr80, sample_nssr90 = evaluator.cal_all_blosum_nssr(
                    sample_logits, sample_seq)
                nssr42.append(sample_nssr42)
                nssr62.append(sample_nssr62)
                nssr80.append(sample_nssr80)
                nssr90.append(sample_nssr90)
            
            # Clear GPU cache more frequently
            if device.type == 'cuda' and (batch_idx + 1) % cache_clear_freq == 0:
                torch.cuda.empty_cache()
    
    # Compute final metrics
    results = metrics.compute()
    
    # Add BLOSUM metrics if available
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
        mp.set_start_method('spawn', force=True)
        # Multi-GPU evaluation
        evaluate_multi_gpu_main(cfg, output_dir, experiment, num_gpus)
    else:
        # Single GPU evaluation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        evaluate_single_gpu_main(cfg, output_dir, experiment, device)
    
    if experiment:
        experiment.end()


def evaluate_single_gpu_main(cfg, output_dir, experiment, device):
    """Main function for single GPU evaluation."""
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
    
    # Initialize model
    print("\nInitializing models...")
    model, device = initialize_model(cfg, device)
    
    # Evaluate each dataset
    all_results = {}
    collator = CollatorDiff()
    
    for dataset_name, dataset in datasets_to_eval.items():
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.evaluation.batch_size, 
            shuffle=False,
            num_workers=cfg.evaluation.num_workers, 
            collate_fn=collator,
            pin_memory=True
        )
        
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
    
    # Save results
    save_evaluation_results(cfg, all_results, output_dir)


def evaluate_multi_gpu_main(cfg, output_dir, experiment, num_gpus):
    """Main function for multi-GPU evaluation using improved strategy."""
    world_size = min(num_gpus, cfg.evaluation.get('max_gpus', num_gpus))
    
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
    
    all_results = {}
    
    for dataset_name, dataset in datasets_to_eval.items():
        print(f"\nEvaluating {dataset_name} dataset with {world_size} GPUs...")
        start_time = time.time()
        
        # Use multiprocessing queue for results
        result_queue = mp.Queue()
        
        # Spawn worker processes
        mp.spawn(
            worker_fn_improved,
            args=(world_size, cfg, dataset, dataset_name, result_queue),
            nprocs=world_size,
            join=True
        )
        
        # Get aggregated results
        results = result_queue.get()
        elapsed_time = time.time() - start_time
        
        results['evaluation_time_seconds'] = elapsed_time
        results['evaluation_time_minutes'] = elapsed_time / 60
        results['num_gpus_used'] = world_size
        all_results[dataset_name] = results
        
        # Create and print table
        table = create_results_table(results, dataset_name.capitalize() + " Set")
        print(f"\n{dataset_name.capitalize()} Set Results:")
        print(table)
        print(f"Evaluation time: {elapsed_time/60:.2f} minutes ({world_size} GPUs)")
        
        # Log to Comet ML if enabled
        if experiment:
            for metric, value in results.items():
                experiment.log_metric(f"{dataset_name}_{metric}", value)
    
    # Save results
    save_evaluation_results(cfg, all_results, output_dir)


def worker_fn_improved(rank, world_size, cfg, dataset, dataset_name, result_queue):
    """Improved worker function using standard data parallelism."""
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
    
    # Create distributed sampler for the original dataset
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    collator = CollatorDiff()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.evaluation.get('batch_size_per_gpu', cfg.evaluation.batch_size),
        sampler=sampler,
        num_workers=cfg.evaluation.num_workers // world_size,
        collate_fn=collator,
        pin_memory=True
    )
    
    # Initialize streaming metrics and evaluator
    metrics = StreamingMetrics(device=device, distributed=True)
    evaluator = Evaluator()
    
    # BLOSUM tracking
    local_nssr42, local_nssr62, local_nssr80, local_nssr90 = [], [], [], []
    
    # Get optimization settings
    use_amp = cfg.evaluation.use_amp and device.type == 'cuda'
    ensemble_num = cfg.evaluation.ensemble_num
    cache_clear_freq = 5 if ensemble_num > 20 else 10
    
    with torch.no_grad():
        for batch_idx, (g_batch, ipa_batch) in enumerate(
            tqdm(dataloader, desc=f"GPU {rank}", disable=rank != 0)
        ):
            g_batch = g_batch.to(device)
            ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
            
            # Process all ensemble predictions for this batch
            ensemble_logits = vectorized_ensemble_sample(
                model, g_batch, ipa_batch, ensemble_num, cfg, use_amp
            )
            
            # Compute mean logits
            mean_logits = ensemble_logits.mean(dim=0)
            del ensemble_logits
            
            # Update streaming metrics
            metrics.update(mean_logits, g_batch.x)
            
            # Calculate BLOSUM metrics if needed
            if evaluator.blosum_eval:
                for i in range(g_batch.x.shape[0]):
                    sample_logits = mean_logits[i].argmax(dim=0)
                    sample_seq = g_batch.x[i].argmax(dim=0)
                    s42, s62, s80, s90 = evaluator.cal_all_blosum_nssr(
                        sample_logits.unsqueeze(0), sample_seq.unsqueeze(0))
                    local_nssr42.append(s42)
                    local_nssr62.append(s62)
                    local_nssr80.append(s80)
                    local_nssr90.append(s90)
            
            # Clear cache periodically
            if (batch_idx + 1) % cache_clear_freq == 0:
                torch.cuda.empty_cache()
    
    # Compute metrics (will be reduced across all processes)
    results = metrics.compute()
    
    # Aggregate BLOSUM metrics if available
    if evaluator.blosum_eval and rank == 0:
        # Gather BLOSUM metrics from all processes
        all_nssr42 = [None] * world_size if rank == 0 else None
        all_nssr62 = [None] * world_size if rank == 0 else None
        all_nssr80 = [None] * world_size if rank == 0 else None
        all_nssr90 = [None] * world_size if rank == 0 else None
        
        dist.gather_object(local_nssr42, all_nssr42, dst=0)
        dist.gather_object(local_nssr62, all_nssr62, dst=0)
        dist.gather_object(local_nssr80, all_nssr80, dst=0)
        dist.gather_object(local_nssr90, all_nssr90, dst=0)
        
        if rank == 0:
            # Flatten lists
            nssr42 = [item for sublist in all_nssr42 for item in sublist]
            nssr62 = [item for sublist in all_nssr62 for item in sublist]
            nssr80 = [item for sublist in all_nssr80 for item in sublist]
            nssr90 = [item for sublist in all_nssr90 for item in sublist]
            
            # Calculate statistics
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
    
    # Send results to main process
    if rank == 0:
        result_queue.put(results)
    
    dist.destroy_process_group()


def save_evaluation_results(cfg, all_results, output_dir):
    """Save evaluation results in various formats."""
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


def initialize_model(cfg, device=None):
    """Initialize and load the model with optimizations."""
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
    use_compile = cfg.evaluation.get('use_compile', True)  # Default to True
    if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile for faster inference...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")
    
    return model, device


if __name__ == "__main__":
    main()