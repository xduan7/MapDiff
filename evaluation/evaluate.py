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
from datetime import datetime, timedelta
import time
import torch.cuda.amp as amp
import pickle
import hashlib
from pathlib import Path
# Import TorchMetrics for distributed evaluation
try:
    import torchmetrics
    from torchmetrics import Accuracy, MeanMetric
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    # Only print warning once
    import os
    if os.environ.get('LOCAL_RANK', '0') == '0' and os.environ.get('RANK', '0') == '0':
        print("Warning: TorchMetrics not available. Using fallback implementation.")


class CathWithID(Dataset):
    """Wrapper for Cath dataset that returns protein IDs along with data."""
    def __init__(self, cath_dataset, protein_ids):
        self.cath_dataset = cath_dataset
        self.protein_ids = protein_ids
    
    def __len__(self):
        return len(self.cath_dataset)
    
    def __getitem__(self, idx):
        return self.cath_dataset[idx], self.protein_ids[idx]


def collate_with_ids(batch, original_collator):
    """Custom collate function that separates protein IDs from data."""
    data_list = []
    protein_ids = []
    
    for data, protein_id in batch:
        data_list.append(data)
        protein_ids.append(protein_id)
    
    # Use original collator for data
    g_batch, ipa_batch = original_collator(data_list)
    
    return (g_batch, ipa_batch), protein_ids


def cal_stats_metric(values):
    mean_value = np.mean(values)
    median_value = np.median(values)
    std_value = np.std(values)
    return mean_value, median_value, std_value


def generate_evaluation_hash(cfg):
    """Generate stable hash from config and script for directory naming."""
    script_path = os.path.abspath(__file__)
    
    # Build hash content with key evaluation parameters
    hash_content = {
        'checkpoint': cfg.evaluation.checkpoint_path,
        'batch_size': cfg.evaluation.get('batch_size_per_gpu', 1),
        'ensemble_num': cfg.evaluation.ensemble_num,
        'ddim_steps': cfg.evaluation.ddim_steps,
        'script_mtime': os.path.getmtime(script_path),
        'script_size': os.path.getsize(script_path),
        # Model config
        'model': OmegaConf.to_container(cfg.model),
        'diffusion': OmegaConf.to_container(cfg.diffusion),
        'mask_prior': OmegaConf.to_container(cfg.mask_prior)
    }
    
    # Create stable hash
    hash_str = json.dumps(hash_content, sort_keys=True)
    full_hash = hashlib.sha256(hash_str.encode()).hexdigest()
    
    return full_hash[:16], full_hash


def save_protein_result(result_path, result_data):
    """Save individual protein result."""
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)


def load_existing_results(results_dir):
    """Load already evaluated protein results."""
    processed_proteins = set()
    results_data = {}
    
    if not os.path.exists(results_dir):
        return processed_proteins, results_data
    
    # Walk through all result files
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_result.json'):
                protein_id = file.replace('_result.json', '')
                processed_proteins.add(protein_id)
                
                # Load result data
                with open(os.path.join(root, file), 'r') as f:
                    results_data[protein_id] = json.load(f)
    
    return processed_proteins, results_data


def save_checkpoint(checkpoint_path, batch_idx, metrics_state, blosum_scores):
    """Save evaluation checkpoint for resuming."""
    checkpoint = {
        'batch_idx': batch_idx,
        'metrics_state': metrics_state,
        'blosum_scores': blosum_scores,
        'timestamp': datetime.now().isoformat()
    }
    # Save to temp file first then rename for atomicity
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    os.rename(temp_path, checkpoint_path)
    

def load_checkpoint(checkpoint_path):
    """Load evaluation checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Resuming from checkpoint: batch {checkpoint['batch_idx']}")
        return checkpoint
    return None


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
    
    def get_state(self):
        """Get current state for checkpointing."""
        return {
            'total_correct': self.total_correct.cpu(),
            'total_samples': self.total_samples.cpu(),
            'total_log_probs': self.total_log_probs.cpu(),
            'per_sample_recoveries': self.per_sample_recoveries.copy()
        }
    
    def load_state(self, state):
        """Load state from checkpoint."""
        self.total_correct = state['total_correct'].to(self.device)
        self.total_samples = state['total_samples'].to(self.device)
        self.total_log_probs = state['total_log_probs'].to(self.device)
        self.per_sample_recoveries = state['per_sample_recoveries'].copy()
    
    def update(self, logits, g_batch):
        """Update metrics with a batch of predictions."""
        targets = g_batch.x
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
        
        # Calculate per-protein recovery
        correct_per_residue = (preds == targets_idx).float()
        for i in range(len(g_batch.ptr) - 1):
            start, end = g_batch.ptr[i], g_batch.ptr[i+1]
            protein_recovery = correct_per_residue[start:end].mean().item()
            self.per_sample_recoveries.append(protein_recovery)

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
            
            # Gather per_sample_recoveries from all ranks
            world_size = dist.get_world_size()
            gathered_recoveries = [None] * world_size
            dist.all_gather_object(gathered_recoveries, self.per_sample_recoveries)
            # Flatten the list of lists into a single list
            all_recoveries = [item for sublist in gathered_recoveries for item in sublist]
        else:
            all_recoveries = self.per_sample_recoveries
        
        full_recovery = (self.total_correct.float() / self.total_samples.float()).item()
        perplexity = torch.exp(-self.total_log_probs / self.total_samples).item()
        
        # Calculate stats from all aggregated recoveries
        if all_recoveries:
            mean_recovery, median_recovery, std_recovery = cal_stats_metric(all_recoveries)
        else:
            mean_recovery, median_recovery, std_recovery = 0.0, 0.0, 0.0
        
        results = {
            'mean_recovery': mean_recovery,
            'median_recovery': median_recovery,
            'std_recovery': std_recovery,
            'full_recovery': full_recovery,
            'perplexity': perplexity,
            'num_samples': self.total_samples.item()  # Use aggregated total_samples
        }
        
        return results


def load_and_filter_protein_ids(dataset_name, protein_ids, results_dir, resume_from_dir=None):
    """Loads existing results and filters protein IDs to find what's left to evaluate."""
    processed_proteins = set()
    existing_results = {}

    # Load from resume directory first
    if resume_from_dir:
        resume_dataset_dir = os.path.join(resume_from_dir, dataset_name.lower().replace(" ", "_"))
        if os.path.exists(resume_dataset_dir):
            resumed_p, resumed_r = load_existing_results(resume_dataset_dir)
            processed_proteins.update(resumed_p)
            existing_results.update(resumed_r)
            print(f"Resuming from {resume_dataset_dir}: Found {len(resumed_p)} previously evaluated proteins.")

    # Load from current results directory and merge
    current_dataset_dir = os.path.join(results_dir, dataset_name.lower().replace(" ", "_"))
    os.makedirs(current_dataset_dir, exist_ok=True)
    current_p, current_r = load_existing_results(current_dataset_dir)
    
    newly_processed_count = len(current_p - processed_proteins)
    if newly_processed_count > 0:
        print(f"Found {newly_processed_count} additional evaluated proteins in the current results directory.")

    processed_proteins.update(current_p)
    existing_results.update(current_r)

    # Filter out processed proteins
    remaining_protein_ids = [pid for pid in protein_ids if pid not in processed_proteins]
    
    print(f"Total unique proteins already processed: {len(processed_proteins)}")
    print(f"Remaining proteins to evaluate: {len(remaining_protein_ids)}")

    return remaining_protein_ids, list(existing_results.values())


def vectorized_ensemble_sample(model, g_batch, ipa_batch, ensemble_num, cfg, use_amp=False):
    """Use torch.vmap for efficient ensemble predictions if available."""
    # Disable vmap for now - it doesn't work well with complex models
    use_vmap = False
    
    if use_vmap and hasattr(torch, 'vmap') and hasattr(torch, 'func'):
        try:
            # Create a function that runs a single ensemble member
            def single_ensemble(dummy):
                return model.mc_ddim_sample(g_batch, ipa_batch, 
                                          diverse=True, 
                                          step=cfg.evaluation.ddim_steps)
            
            # Vectorize over ensemble dimension
            ensemble_fn = torch.vmap(single_ensemble, in_dims=0, out_dims=0)
            dummy_input = torch.zeros(ensemble_num)
            all_logits, _ = ensemble_fn(dummy_input)
            return all_logits
        except Exception as e:
            # Fallback to sequential if vmap fails
            pass
    
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


def evaluate_dataset(model, dataloader, evaluator, device, cfg, dataset_name="Dataset", results_dir=None, protein_ids=None, resume_from_dir=None):
    """Evaluate model on a dataset with protein-level checkpointing."""
    model.eval()
    enable_dropout(model)
    
    # If results_dir is provided, check for existing results
    protein_results = []
    dataset_results_dir = None
    
    if results_dir and protein_ids:
        dataset_results_dir = os.path.join(results_dir, dataset_name.lower().replace(" ", "_"))
        os.makedirs(dataset_results_dir, exist_ok=True)
        
        # This function now handles both resume and current directories
        remaining_protein_ids, protein_results = load_and_filter_protein_ids(
            dataset_name, protein_ids, results_dir, resume_from_dir
        )

        if not remaining_protein_ids:
            print(f"All proteins already evaluated for {dataset_name}")
            return aggregate_protein_results(protein_results)

        # Filter the dataset to only include remaining proteins
        # This is a simplification; in a real scenario, you might need to re-create the dataloader
        # For this example, we assume the dataloader will still iterate through all, and we will skip inside the loop
        processed_proteins = set(protein_ids) - set(remaining_protein_ids)
    else:
        processed_proteins = set()
    
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
    
    # Track batch offset for protein IDs
    batch_offset = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}", 
                                      disable=not cfg.logging.verbose)):
            # Handle both regular and ID-tracked dataloaders
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                (g_batch, ipa_batch), batch_protein_ids = batch_data
            else:
                g_batch, ipa_batch = batch_data
                batch_protein_ids = None
            
            g_batch = g_batch.to(device)
            ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
            
            # Skip if all proteins in batch are already processed
            if batch_protein_ids and dataset_results_dir:
                skip_indices = [i for i, pid in enumerate(batch_protein_ids) if pid in processed_proteins]
                if len(skip_indices) == len(batch_protein_ids):
                    batch_offset += len(batch_protein_ids)
                    continue
            
            # Get ensemble predictions
            ensemble_logits = vectorized_ensemble_sample(
                model, g_batch, ipa_batch, ensemble_num, cfg, use_amp
            )
            
            # Compute mean logits
            mean_logits = ensemble_logits.mean(dim=0)
            
            # Free ensemble memory immediately
            del ensemble_logits
            
            # Update streaming metrics with batch
            metrics.update(mean_logits, g_batch)
            
            # Process per-protein results using ptr boundaries (FIXED!)
            preds = mean_logits.argmax(dim=-1)
            targets_idx = g_batch.x.argmax(dim=-1)
            
            # Iterate over proteins using ptr boundaries
            num_proteins = len(g_batch.ptr) - 1
            for i in range(num_proteins):
                start, end = g_batch.ptr[i], g_batch.ptr[i+1]
                
                # Skip if already processed
                if batch_protein_ids and batch_protein_ids[i] in processed_proteins:
                    continue
                
                # Calculate metrics for this protein
                pred_protein = preds[start:end]
                target_protein = targets_idx[start:end]
                protein_correct = (pred_protein == target_protein).sum()
                protein_samples = end - start
                protein_recovery = (protein_correct / protein_samples).item()
                
                # Calculate log probabilities
                log_probs = F.log_softmax(mean_logits[start:end], dim=-1)
                target_log_probs = torch.gather(log_probs, -1, target_protein.unsqueeze(-1)).squeeze(-1)
                protein_log_prob = target_log_probs.sum().item()
                
                protein_result = {
                    'recovery': protein_recovery,
                    'correct': protein_correct.item(),
                    'samples': protein_samples,
                    'log_probs': protein_log_prob
                }
                
                if batch_protein_ids and dataset_results_dir:
                    protein_id = batch_protein_ids[i]
                    result_path = os.path.join(dataset_results_dir, f"{protein_id}_result.json")
                    save_protein_result(result_path, protein_result)
                    processed_proteins.add(protein_id)
                    
                    # LOG PROGRESS SO WE KNOW WTF IS HAPPENING
                    if len(processed_proteins) % 100 == 0:
                        print(f"[PROGRESS] Saved {len(processed_proteins)} proteins to {dataset_results_dir}")
                
                protein_results.append(protein_result)
            
            # Calculate BLOSUM metrics if needed
            if evaluator.blosum_eval:
                pred_seqs = mean_logits.argmax(dim=-1)
                target_seqs = g_batch.x.argmax(dim=-1)
                for i in range(len(g_batch.ptr) - 1):
                    start, end = g_batch.ptr[i], g_batch.ptr[i+1]
                    pred_protein_seq = pred_seqs[start:end]
                    target_protein_seq = target_seqs[start:end]
                    s42, s62, s80, s90 = evaluator.cal_all_blosum_nssr(
                        pred_protein_seq, target_protein_seq)
                    nssr42.append(s42)
                    nssr62.append(s62)
                    nssr80.append(s80)
                    nssr90.append(s90)
            
            batch_offset += num_proteins
            
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


def aggregate_protein_results(protein_results):
    """Aggregate protein-level results into dataset-level metrics."""
    all_recoveries = []
    total_correct = 0
    total_samples = 0
    total_log_probs = 0.0
    
    for result in protein_results:
        all_recoveries.append(result['recovery'])
        total_correct += result['correct']
        total_samples += result['samples']
        total_log_probs += result['log_probs']
    
    # Calculate aggregate metrics
    mean_recovery, median_recovery, std_recovery = cal_stats_metric(all_recoveries)
    full_recovery = total_correct / total_samples if total_samples > 0 else 0.0
    perplexity = torch.exp(torch.tensor(-total_log_probs / total_samples)).item() if total_samples > 0 else float('inf')
    
    return {
        'mean_recovery': mean_recovery,
        'median_recovery': median_recovery,
        'std_recovery': std_recovery,
        'full_recovery': full_recovery,
        'perplexity': perplexity,
        'num_samples': total_samples
    }


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
    # Generate hash for results directory
    hash_short, hash_full = generate_evaluation_hash(cfg)
    results_base_dir = os.path.join('./results', hash_short)
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Check for resume directory
    resume_from_dir = cfg.evaluation.get('resume_from_dir', None)
    
    print("="*80)
    print(f"MapDiff Checkpoint Evaluation")
    print(f"Results directory: {results_base_dir}")
    print(f"Hash: {hash_short}")
    print(f"Checkpoint: {cfg.evaluation.checkpoint_path}")
    if resume_from_dir:
        print(f"Resuming from: {resume_from_dir}")
    print("="*80)
    
    # Save configuration and hash info
    config_path = os.path.join(results_base_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    hash_info = {
        'hash_short': hash_short,
        'hash_full': hash_full,
        'timestamp': datetime.now().isoformat(),
        'script': os.path.abspath(__file__)
    }
    with open(os.path.join(results_base_dir, 'hash_info.json'), 'w') as f:
        json.dump(hash_info, f, indent=2)
    
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
        experiment.set_name(f"eval_{hash_short}")
        if cfg.comet.comet_tag:
            experiment.add_tag(cfg.comet.comet_tag)
        experiment.add_tag("evaluation")
    
    # Get Hydra output directory for compatibility
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Check if multi-GPU is requested and available
    use_multi_gpu = hasattr(cfg.evaluation, 'use_multi_gpu') and cfg.evaluation.use_multi_gpu
    num_gpus = torch.cuda.device_count() if use_multi_gpu else 1
    
    if use_multi_gpu and num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel evaluation")
        mp.set_start_method('spawn', force=True)
        # Multi-GPU evaluation
        evaluate_multi_gpu_main(cfg, results_base_dir, experiment, num_gpus, resume_from_dir)
    else:
        # Single GPU evaluation
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        evaluate_single_gpu_main(cfg, results_base_dir, experiment, device, resume_from_dir)
    
    if experiment:
        experiment.end()


def evaluate_single_gpu_main(cfg, output_dir, experiment, device, resume_from_dir=None):
    """Main function for single GPU evaluation."""
    set_seed()
    
    # Load datasets
    print("\nLoading datasets...")
    datasets_to_eval = {}
    protein_ids_map = {}
    
    if cfg.evaluation.evaluate_val:
        val_ID = os.listdir(cfg.dataset.val_dir)
        val_dataset = Cath(val_ID, cfg.dataset.val_dir)
        datasets_to_eval['validation'] = val_dataset
        protein_ids_map['validation'] = val_ID
        print(f"Validation dataset size: {len(val_dataset)}")
    
    if cfg.evaluation.evaluate_test:
        test_ID = os.listdir(cfg.dataset.test_dir)
        test_dataset = Cath(test_ID, cfg.dataset.test_dir)
        datasets_to_eval['test'] = test_dataset
        protein_ids_map['test'] = test_ID
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
        # Create dataset with IDs
        protein_ids = protein_ids_map[dataset_name]
        dataset_with_ids = CathWithID(dataset, protein_ids)
        
        # Create custom collate function
        def collate_fn(batch):
            return collate_with_ids(batch, collator)
        
        dataloader = DataLoader(
            dataset_with_ids, 
            batch_size=cfg.evaluation.batch_size, 
            shuffle=False,
            num_workers=cfg.evaluation.num_workers, 
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        start_time = time.time()
        results = evaluate_dataset(model, dataloader, evaluator, device, cfg, 
                                 dataset_name.capitalize() + " Set", 
                                 results_dir=output_dir,
                                 protein_ids=protein_ids,
                                 resume_from_dir=resume_from_dir)
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


def evaluate_multi_gpu_main(cfg, output_dir, experiment, num_gpus, resume_from_dir=None):
    """Main function for multi-GPU evaluation using improved strategy."""
    world_size = min(num_gpus, cfg.evaluation.get('max_gpus', num_gpus))

    def find_free_port():
        import socket
        from contextlib import closing
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    # Find a free port and update config
    if world_size > 1:
        master_port = find_free_port()
        OmegaConf.set_struct(cfg, False)
        cfg.evaluation.master_port = master_port
        OmegaConf.set_struct(cfg, True)
        if cfg.logging.get('verbose', True):
            print(f"Using free port {master_port} for distributed evaluation.")
    
    # Load datasets
    print("\nLoading datasets...")
    datasets_to_eval = {}
    protein_ids_map = {}
    
    if cfg.evaluation.evaluate_val:
        val_ID = os.listdir(cfg.dataset.val_dir)
        val_dataset = Cath(val_ID, cfg.dataset.val_dir)
        datasets_to_eval['validation'] = val_dataset
        protein_ids_map['validation'] = val_ID
        print(f"Validation dataset size: {len(val_dataset)}")
    
    if cfg.evaluation.evaluate_test:
        test_ID = os.listdir(cfg.dataset.test_dir)
        test_dataset = Cath(test_ID, cfg.dataset.test_dir)
        datasets_to_eval['test'] = test_dataset
        protein_ids_map['test'] = test_ID
        print(f"Test dataset size: {len(test_dataset)}")
    
    all_results = {}
    
    for dataset_name, full_dataset in datasets_to_eval.items():
        print(f"\nEvaluating {dataset_name} dataset with {world_size} GPUs...")
        
        dataset_results_dir = os.path.join(output_dir, dataset_name.lower().replace(" ", "_"))
        protein_ids = protein_ids_map[dataset_name]

        remaining_protein_ids, existing_results = load_and_filter_protein_ids(
            dataset_name, protein_ids, output_dir, resume_from_dir
        )

        if not remaining_protein_ids:
            print(f"All proteins already evaluated for {dataset_name}")
            results = aggregate_protein_results(existing_results)
            elapsed_time = 0
        else:
            start_time = time.time()
            
            # Create a new dataset with only the remaining proteins
            if dataset_name == 'validation':
                dataset_path = cfg.dataset.val_dir
            else:
                dataset_path = cfg.dataset.test_dir
            
            todo_dataset = Cath(remaining_protein_ids, dataset_path)
            todo_dataset_with_ids = CathWithID(todo_dataset, remaining_protein_ids)

            # Use multiprocessing queue for results
            result_queue = mp.Queue()
            
            # Spawn worker processes
            mp.spawn(
                worker_fn_improved,
                args=(world_size, cfg, dataset_name, result_queue, todo_dataset_with_ids, output_dir),
                nprocs=world_size,
                join=True
            )
            
            # Get aggregated metrics for the newly evaluated proteins
            new_metrics = result_queue.get()
            elapsed_time = time.time() - start_time

            # After workers finish, reload all results from disk for final aggregation
            _, all_protein_results_map = load_existing_results(dataset_results_dir)
            results = aggregate_protein_results(list(all_protein_results_map.values()))

            # Update with metrics that are hard to aggregate from files (e.g., BLOSUM)
            for key, value in new_metrics.items():
                if 'nssr' in key:
                    results[key] = value
        
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



def worker_fn_improved(rank, world_size, cfg, dataset_name, result_queue, todo_dataset_with_ids, output_dir):
    """Improved worker function using standard data parallelism."""
    # Set up distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.evaluation.get('master_port', '12355'))
    
    # Set NCCL environment variables for better stability
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # Increase timeout for large-scale multi-GPU setups
    dist.init_process_group(
        backend='nccl', 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )
    dist.barrier()
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    set_seed()

    # The dataset is now pre-filtered and passed in
    dataset_with_ids = todo_dataset_with_ids
    
    # Initialize model on this GPU
    model, _ = initialize_model(cfg, device)
    model.eval()
    enable_dropout(model)
    
    # Create distributed sampler for the dataset WITH IDS
    sampler = DistributedSampler(
        dataset_with_ids, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    collator = CollatorDiff()
    
    # Create partial function for collate_fn to avoid pickling issues
    from functools import partial
    collate_fn = partial(collate_with_ids, original_collator=collator)
    
    dataloader = DataLoader(
        dataset_with_ids,
        batch_size=cfg.evaluation.batch_size,
        sampler=sampler,
        num_workers=max(2, cfg.evaluation.num_workers // world_size),
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Initialize streaming metrics and evaluator
    metrics = StreamingMetrics(device=device, distributed=True)
    evaluator = Evaluator()
    
    # Move BLOSUM matrices to GPU if available
    if evaluator.blosum_eval and hasattr(evaluator, 'blosum_mats'):
        for blosum_name in evaluator.blosum_mats.keys():
            evaluator.blosum_mats[blosum_name] = evaluator.blosum_mats[blosum_name].to(device)
    
    # BLOSUM tracking
    local_nssr42, local_nssr62, local_nssr80, local_nssr90 = [], [], [], []
    
    # Set up for saving protein results - JUST LIKE SINGLE-GPU!
    dataset_results_dir = os.path.join(output_dir, dataset_name.lower().replace(" ", "_"))
    processed_proteins_in_worker = set() # Track proteins processed by this worker
    
    # Get optimization settings
    use_amp = cfg.evaluation.use_amp and device.type == 'cuda'
    ensemble_num = cfg.evaluation.ensemble_num
    cache_clear_freq = 5 if ensemble_num > 20 else 10
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(
            tqdm(dataloader, desc=f"GPU {rank}", disable=rank != 0)
        ):
            # HANDLE THE SAME WAY AS SINGLE-GPU!!!
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                (g_batch, ipa_batch), batch_protein_ids = batch_data
            else:
                g_batch, ipa_batch = batch_data
                batch_protein_ids = None
                
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
            metrics.update(mean_logits, g_batch)
            
            # SAVE PER-PROTEIN RESULTS LIKE SINGLE-GPU DOES!!!
            if batch_protein_ids and dataset_results_dir:
                pred_seqs = mean_logits.argmax(dim=-1)
                target_seqs = g_batch.x.argmax(dim=-1)
                
                # Per-protein metrics using ptr boundaries
                for i in range(len(g_batch.ptr) - 1):
                    start, end = g_batch.ptr[i], g_batch.ptr[i+1]
                    
                    protein_id = batch_protein_ids[i]
                    
                    # Calculate recovery for this protein
                    pred_protein = pred_seqs[start:end]
                    target_protein = target_seqs[start:end]
                    protein_correct = (pred_protein == target_protein).sum()
                    protein_samples = end - start
                    protein_recovery = (protein_correct / protein_samples).item()
                    
                    # Calculate log probabilities
                    log_probs = F.log_softmax(mean_logits[start:end], dim=-1)
                    target_log_probs = torch.gather(log_probs, -1, target_protein.unsqueeze(-1)).squeeze(-1)
                    protein_log_prob = target_log_probs.sum().item()
                    
                    # Save result
                    protein_result = {
                        'recovery': protein_recovery,
                        'correct': protein_correct.item(),
                        'samples': protein_samples,
                        'log_probs': protein_log_prob
                    }
                    
                    result_path = os.path.join(dataset_results_dir, f"{protein_id}_result.json")
                    save_protein_result(result_path, protein_result)
                    processed_proteins_in_worker.add(protein_id)
                    
                    if rank == 0 and len(processed_proteins_in_worker) % 100 == 0:
                        print(f"[PROGRESS] Worker 0 has saved {len(processed_proteins_in_worker)} proteins.")
            
            # Calculate BLOSUM metrics if needed
            if evaluator.blosum_eval:
                if not batch_protein_ids:  # Only compute if not already done above
                    pred_seqs = mean_logits.argmax(dim=-1)
                    target_seqs = g_batch.x.argmax(dim=-1)
                    
                for i in range(len(g_batch.ptr) - 1):
                    start, end = g_batch.ptr[i], g_batch.ptr[i+1]
                    pred_protein_seq = pred_seqs[start:end]
                    target_protein_seq = target_seqs[start:end]
                    s42, s62, s80, s90 = evaluator.cal_all_blosum_nssr(
                        pred_protein_seq, target_protein_seq)
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
    if evaluator.blosum_eval:
        # All ranks must participate in gather_object
        all_nssr42 = [None] * world_size if rank == 0 else None
        all_nssr62 = [None] * world_size if rank == 0 else None
        all_nssr80 = [None] * world_size if rank == 0 else None
        all_nssr90 = [None] * world_size if rank == 0 else None
        
        dist.gather_object(local_nssr42, all_nssr42, dst=0)
        dist.gather_object(local_nssr62, all_nssr62, dst=0)
        dist.gather_object(local_nssr80, all_nssr80, dst=0)
        dist.gather_object(local_nssr90, all_nssr90, dst=0)
        
        # Only rank 0 processes the gathered results
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