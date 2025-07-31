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
import pickle
from pathlib import Path
import queue
import threading


def cal_stats_metric(values):
    mean_value = np.mean(values)
    median_value = np.median(values)
    std_value = np.std(values)
    return mean_value, median_value, std_value


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


def save_checkpoint(checkpoint_path, processed_ids, results_dict):
    """Save evaluation checkpoint for resuming."""
    checkpoint = {
        'processed_ids': processed_ids,
        'results_dict': results_dict,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path):
    """Load evaluation checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint['processed_ids'], checkpoint['results_dict']
    return set(), {}


def ensemble_sample_protein(model, g_batch, ipa_batch, ensemble_num, cfg):
    """Generate ensemble predictions for a single protein."""
    ensemble_logits = []
    
    with torch.no_grad():
        for _ in range(ensemble_num):
            logits, _ = model.mc_ddim_sample(
                g_batch, ipa_batch, 
                diverse=True, 
                step=cfg.evaluation.ddim_steps
            )
            ensemble_logits.append(logits)
    
    # Stack and average ensemble predictions
    ensemble_logits = torch.stack(ensemble_logits)
    mean_logits = ensemble_logits.mean(dim=0)
    
    return mean_logits


def evaluate_single_protein(protein_id, dataset, model, evaluator, cfg, device):
    """Evaluate a single protein with ensemble predictions."""
    # Get protein data
    g, ipa = dataset[dataset.IDs.index(protein_id)]
    
    # Create batch of size 1
    collator = CollatorDiff()
    g_batch, ipa_batch = collator([(g, ipa)])
    
    g_batch = g_batch.to(device)
    ipa_batch = ipa_batch.to(device) if ipa_batch is not None else None
    
    # Generate ensemble predictions
    mean_logits = ensemble_sample_protein(
        model, g_batch, ipa_batch, 
        cfg.evaluation.ensemble_num, cfg
    )
    
    # Calculate metrics
    pred_seq = mean_logits.argmax(dim=1)
    target_seq = g_batch.x.argmax(dim=1)
    
    # Recovery
    recovery = (pred_seq == target_seq).float().mean().item()
    
    # Perplexity - using evaluator's method for consistency with original
    perplexity = evaluator.cal_perplexity(mean_logits, g_batch.x)
    
    result = {
        'protein_id': protein_id,
        'recovery': recovery,
        'perplexity': perplexity,
        'sequence_length': target_seq.shape[0],
        # Store for dataset-level perplexity calculation
        'logits': mean_logits.cpu(),
        'target': g_batch.x.cpu()
    }
    
    # BLOSUM metrics if needed
    if evaluator.blosum_eval:
        nssr_scores = evaluator.cal_all_blosum_nssr(pred_seq, target_seq)
        result['nssr42'] = nssr_scores[0]
        result['nssr62'] = nssr_scores[1]
        result['nssr80'] = nssr_scores[2]
        result['nssr90'] = nssr_scores[3]
    
    return result


def gpu_worker(gpu_id, task_queue, result_queue, cfg, dataset, stop_event):
    """GPU worker that processes protein tasks from queue."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    set_seed()
    
    # Initialize model on this GPU
    model, _ = initialize_model(cfg, device)
    model.eval()
    enable_dropout(model)
    
    # Initialize evaluator
    evaluator = Evaluator()
    if evaluator.blosum_eval and hasattr(evaluator, 'blosum_mats'):
        for blosum_name in evaluator.blosum_mats.keys():
            evaluator.blosum_mats[blosum_name] = evaluator.blosum_mats[blosum_name].to(device)
    
    while not stop_event.is_set():
        try:
            # Get next protein to evaluate
            protein_id = task_queue.get(timeout=1)
            
            # Evaluate protein
            result = evaluate_single_protein(
                protein_id, dataset, model, evaluator, cfg, device
            )
            
            # Send result back
            result_queue.put((protein_id, result))
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"GPU {gpu_id} error processing protein: {e}")
            result_queue.put((protein_id, {'error': str(e)}))


def evaluate_dataset_taskbased(cfg, dataset, dataset_name, output_dir, num_gpus=None):
    """Evaluate dataset using task-based parallelism."""
    if num_gpus is None:
        num_gpus = min(torch.cuda.device_count(), cfg.evaluation.get('max_gpus', 8))
    
    print(f"\nEvaluating {dataset_name} with {num_gpus} GPUs using task-based parallelism")
    print(f"Dataset size: {len(dataset)} proteins")
    
    # Create checkpoint path
    checkpoint_path = os.path.join(output_dir, f"{dataset_name}_checkpoint.pkl")
    
    # Load checkpoint if exists
    processed_ids, results_dict = load_checkpoint(checkpoint_path)
    if processed_ids:
        print(f"Resuming from checkpoint: {len(processed_ids)} proteins already processed")
    
    # Get list of proteins to process
    all_protein_ids = dataset.IDs
    remaining_ids = [pid for pid in all_protein_ids if pid not in processed_ids]
    
    if not remaining_ids:
        print("All proteins already processed!")
        return aggregate_results(results_dict, dataset_name)
    
    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Fill task queue with remaining proteins
    for protein_id in remaining_ids:
        task_queue.put(protein_id)
    
    # Start GPU workers
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, cfg, dataset, stop_event)
        )
        p.start()
        workers.append(p)
    
    # Progress tracking
    start_time = time.time()
    pbar = tqdm(total=len(remaining_ids), desc=f"Evaluating {dataset_name}")
    
    # Collect results
    try:
        while len(processed_ids) < len(all_protein_ids):
            try:
                protein_id, result = result_queue.get(timeout=30)
                
                # Store result
                results_dict[protein_id] = result
                processed_ids.add(protein_id)
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'recovery': f"{result.get('recovery', 0):.3f}",
                    'gpu_util': f"{task_queue.qsize()}/{num_gpus}"
                })
                
                # Save checkpoint after every protein
                save_checkpoint(checkpoint_path, processed_ids, results_dict)
                    
            except queue.Empty:
                # Check if all tasks are done
                if task_queue.empty() and len(processed_ids) == len(all_protein_ids):
                    break
                    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    finally:
        # Stop workers
        stop_event.set()
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # Save final checkpoint
        save_checkpoint(checkpoint_path, processed_ids, results_dict)
        pbar.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessed {len(processed_ids)} proteins in {elapsed_time/60:.2f} minutes")
    
    # Aggregate results
    return aggregate_results(results_dict, dataset_name)


def aggregate_results(results_dict, dataset_name):
    """Aggregate individual protein results into dataset metrics."""
    recoveries = []
    total_correct = 0
    total_samples = 0
    
    nssr42_list = []
    nssr62_list = []
    nssr80_list = []
    nssr90_list = []
    
    # For dataset-level perplexity calculation (matching original)
    all_logits = []
    all_targets = []
    
    for protein_id, result in results_dict.items():
        if 'error' in result:
            continue
            
        recoveries.append(result['recovery'])
        
        # Store logits and targets for dataset-level perplexity
        if 'logits' in result and 'target' in result:
            all_logits.append(result['logits'])
            all_targets.append(result['target'])
        
        # For full recovery calculation
        correct = int(result['recovery'] * result['sequence_length'])
        total_correct += correct
        total_samples += result['sequence_length']
        
        # BLOSUM scores
        if 'nssr42' in result:
            nssr42_list.append(result['nssr42'])
            nssr62_list.append(result['nssr62'])
            nssr80_list.append(result['nssr80'])
            nssr90_list.append(result['nssr90'])
    
    # Calculate aggregate metrics
    mean_recovery, median_recovery, std_recovery = cal_stats_metric(recoveries)
    full_recovery = total_correct / total_samples if total_samples > 0 else 0
    
    # Calculate dataset-level perplexity (matching original implementation)
    if all_logits and all_targets:
        from evaluator import Evaluator
        evaluator = Evaluator()
        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        perplexity = evaluator.cal_perplexity(all_logits_tensor, all_targets_tensor)
    else:
        # Fallback to individual perplexities if logits not stored
        perplexities = [result['perplexity'] for result in results_dict.values() if 'perplexity' in result and 'error' not in result]
        perplexity = np.mean(perplexities) if perplexities else 0.0
    
    results = {
        'mean_recovery': mean_recovery,
        'median_recovery': median_recovery,
        'std_recovery': std_recovery,
        'full_recovery': full_recovery,
        'perplexity': mean_perplexity,
        'num_samples': len(recoveries)
    }
    
    # Add BLOSUM metrics if available
    if nssr42_list:
        mean_nssr42, median_nssr42, std_nssr42 = cal_stats_metric(nssr42_list)
        mean_nssr62, median_nssr62, std_nssr62 = cal_stats_metric(nssr62_list)
        mean_nssr80, median_nssr80, std_nssr80 = cal_stats_metric(nssr80_list)
        mean_nssr90, median_nssr90, std_nssr90 = cal_stats_metric(nssr90_list)
        
        results.update({
            'mean_nssr42': mean_nssr42, 'median_nssr42': median_nssr42, 'std_nssr42': std_nssr42,
            'mean_nssr62': mean_nssr62, 'median_nssr62': median_nssr62, 'std_nssr62': std_nssr62,
            'mean_nssr80': mean_nssr80, 'median_nssr80': median_nssr80, 'std_nssr80': std_nssr80,
            'mean_nssr90': mean_nssr90, 'median_nssr90': median_nssr90, 'std_nssr90': std_nssr90
        })
    
    return results


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
    use_compile = cfg.evaluation.get('use_compile', True)
    if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile for faster inference...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")
    
    return model, device


def save_evaluation_results(cfg, all_results, output_dir):
    """Save evaluation results in various formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save pretty tables
    tables_path = os.path.join(output_dir, "evaluation_tables.txt")
    with open(tables_path, "w") as f:
        f.write(f"MapDiff Evaluation Results (Task-Based)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {cfg.evaluation.checkpoint_path}\n")
        f.write("="*80 + "\n\n")
        
        for dataset_name, results in all_results.items():
            table = create_results_table(results, dataset_name.capitalize() + " Set")
            f.write(f"{dataset_name.capitalize()} Set Results:\n")
            f.write(table.get_string() + "\n\n")
    
    print(f"\nTables saved to: {tables_path}")
    
    # Save as JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    json_data = {
        'timestamp': timestamp,
        'checkpoint': cfg.evaluation.checkpoint_path,
        'config': OmegaConf.to_container(cfg),
        'results': all_results
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON results saved to: {json_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate_checkpoint")
def main(cfg: DictConfig):
    # Get output directory from Hydra
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    print("="*80)
    print(f"MapDiff Checkpoint Evaluation (Task-Based)")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {cfg.evaluation.checkpoint_path}")
    print("="*80)
    
    # Check if multi-GPU is available
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available! Exiting.")
        return
    
    use_multi_gpu = cfg.evaluation.get('use_multi_gpu', True)
    if not use_multi_gpu:
        num_gpus = 1
    
    print(f"Using {num_gpus} GPUs for task-based evaluation")
    
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
    
    # Evaluate each dataset
    mp.set_start_method('spawn', force=True)
    all_results = {}
    
    for dataset_name, dataset in datasets_to_eval.items():
        results = evaluate_dataset_taskbased(
            cfg, dataset, dataset_name, output_dir, num_gpus
        )
        all_results[dataset_name] = results
        
        # Print results in original format
        print(f"\n{dataset_name} median recovery rate is {results['median_recovery']:.4f}")
        print(f"{dataset_name} perplexity is: {results['perplexity']:.4f}")
        
        # Also print table for detailed view
        table = create_results_table(results, dataset_name.capitalize() + " Set")
        print(f"\n{dataset_name.capitalize()} Set Results (Detailed):")
        print(table)
    
    # Save all results
    save_evaluation_results(cfg, all_results, output_dir)


if __name__ == "__main__":
    main()