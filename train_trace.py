"""
HF_ENDPOINT=https://hf-mirror.com python train_trace.py \
    --data_path ./data/trace \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --num_tasks 5 \
    --output_dir ./outputs
"""
import os
import sys
import json
import argparse
import logging
import csv
from datetime import datetime
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tgr_lora import (
    TGRLoRAConfig, TaskConfig,
    TGRLoRAModel, TGRLoRATrainer, MemoryBank,
    TRACEDataModule, TRACEEvaluator, TRACE_TASKS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TGR-LoRA TRACE Training")
    
    parser.add_argument("--data_path", type=str, default="./data/trace")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=300)
    
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    parser.add_argument("--probe_steps", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=5e-5)
    parser.add_argument("--similarity_threshold_in", type=float, default=0.35)
    parser.add_argument("--similarity_threshold_out", type=float, default=0.35)
    parser.add_argument("--boost_factor", type=float, default=0.5)
    parser.add_argument("--damp_factor", type=float, default=1.0)
    
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    return parser.parse_args()


def compute_cl_metrics(matrix: np.ndarray, current_idx: int) -> dict:
    k = current_idx + 1
    acc = np.mean(matrix[current_idx, :k])
    bwt = 0.0
    if k > 1:
        for j in range(k - 1):
            bwt += matrix[current_idx, j] - matrix[j, j]
        bwt /= (k - 1)
    
    forgetting = 0.0
    if k > 1:
        for j in range(k - 1):
            max_perf = np.max(matrix[:current_idx+1, j])
            forgetting += max(0, max_perf - matrix[current_idx, j])
        forgetting /= (k - 1)
    
    return {
        "average_accuracy": acc,
        "backward_transfer": bwt,
        "forgetting": forgetting
    }


def print_matrix(matrix: np.ndarray, task_ids: list, title: str):
    print(f"\n{'='*20} {title} {'='*20}")
    header = f"{'After':<12}|" + "|".join([f"{t[:8]:^10}" for t in task_ids])
    print(header)
    print("-" * len(header))
    
    for i, tid in enumerate(task_ids):
        row = f"{tid[:12]:<12}|"
        for j in range(len(task_ids)):
            if matrix[i, j] > 0:
                row += f"{matrix[i, j]:^10.4f}|"
            else:
                row += f"{'---':^10}|"
        print(row)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to {seed}")

def main():
    args = parse_args()
    set_seed(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        print("NO PADDING TOKEN!")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    config = TGRLoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        probe_steps=args.probe_steps,
        probe_lr=args.probe_lr,
        similarity_threshold_in=args.similarity_threshold_in,
        similarity_threshold_out=args.similarity_threshold_out,
        boost_factor=args.boost_factor,
        damp_factor=args.damp_factor,
        memory_bank_path=os.path.join(output_dir, "memory_bank"),
    )
    
    tgr_model = TGRLoRAModel(model, config)
    tgr_model.freeze_base_model()
    tgr_model.print_trainable_parameters()
    
    memory_bank = MemoryBank(config)
    trainer = TGRLoRATrainer(tgr_model, config, memory_bank, device)
    
    data_module = TRACEDataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        test_samples=args.test_samples,
    )
    
    evaluator = TRACEEvaluator(tgr_model, tokenizer, device)
    
    task_ids = list(TRACE_TASKS.keys())[:args.num_tasks]
    n_tasks = len(task_ids)
    logger.info(f"Tasks: {task_ids}")
    
    test_matrix = np.zeros((n_tasks, n_tasks))
    all_test_loaders = {}
    
    for task_idx, task_id in enumerate(task_ids):
        logger.info(f"Task {task_idx + 1}/{n_tasks}: {task_id}")

        # load data
        data_module.load_task(task_id)
        train_loader = data_module.get_train_dataloader(task_id)
        test_loader = data_module.get_eval_dataloader(task_id, split='test')
        all_test_loaders[task_id] = test_loader
        
        # training
        task_config = TaskConfig(
            task_id=task_id,
            task_name=TRACE_TASKS[task_id].task_name,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        
        result = trainer.learn_task(
            task_id=task_id,
            train_dataloader=train_loader,
            task_config=task_config,
        )
        
        logger.info(f"Final Loss: {result.final_loss:.4f}")
        logger.info(f"Similar Tasks: {result.similar_tasks}")
        logger.info(f"Different Tasks: {result.different_tasks}")
        
        # evaluate
        logger.info(f"\nEvaluating on TEST sets (tasks 1-{task_idx+1})...")
        
        for eval_idx in range(task_idx + 1):
            eval_task_id = task_ids[eval_idx]
            metrics = evaluator.evaluate_task(
                eval_task_id,
                all_test_loaders[eval_task_id],
                max_new_tokens=args.max_new_tokens
            )
            
            score = metrics.accuracy if metrics.accuracy > 0 else metrics.rougeL
            test_matrix[task_idx, eval_idx] = score
            logger.info(f"  {eval_task_id}: {score:.4f}")
        
        print_matrix(test_matrix, task_ids[:task_idx+1], f"After Task {task_idx+1}")
        
        cl_metrics = compute_cl_metrics(test_matrix, task_idx)
        print(f"\nCL Metrics after task {task_idx+1}:")
        print(f"  Average Accuracy:  {cl_metrics['average_accuracy']:.4f}")
        print(f"  Backward Transfer: {cl_metrics['backward_transfer']:.4f}")
        print(f"  Forgetting:        {cl_metrics['forgetting']:.4f}")
        
        with open(os.path.join(output_dir, f"result_task_{task_idx+1}.json"), "w") as f:
            json.dump({
                "task_idx": task_idx,
                "task_id": task_id,
                "test_matrix": test_matrix.tolist(),
                "cl_metrics": cl_metrics,
            }, f, indent=2)
    
    logger.info("Training Complete!")
    
    print_matrix(test_matrix, task_ids, "FINAL")
    final_metrics = compute_cl_metrics(test_matrix, n_tasks - 1)
    
    print(f"\nFINAL CL Metrics:")
    print(f"Average Accuracy: {final_metrics['average_accuracy']:.4f}")
    print(f"Backward Transfer: {final_metrics['backward_transfer']:.4f}")
    print(f"Forgetting: {final_metrics['forgetting']:.4f}")
    
    final_summary = {
        "config": vars(args),
        "task_order": task_ids,
        "test_matrix": test_matrix.tolist(),
        "final_metrics": final_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(os.path.join(output_dir, "final_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, "test_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["After\\On"] + task_ids)
        for i, tid in enumerate(task_ids):
            writer.writerow([tid] + [f"{test_matrix[i,j]:.4f}" for j in range(n_tasks)])
    
    logger.info(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()