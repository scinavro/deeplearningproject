import os
import json
import random
import time, datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from arc import ARCSolver
from transformers import set_seed
from datasets import Dataset
from zoneinfo import ZoneInfo

from utils import split_examples, plot_arc_example

AUTO_EVAL = True

MAX_DATA = 100  # number of tasks to evaluate (1 set eval / 1 task)
CHECKPOINT_BASE_DIR = "checkpoints/task-cycle-with-epoch-3"
VISUALIZE = False

def check_match_and_pixel_accuracy(pred, truth):
    pred = np.array(pred, dtype=np.uint8)
    truth = np.array(truth, dtype=np.uint8)
    if len(pred.shape) != 2 or pred.shape != truth.shape:
        return 0, 0.0
    match = int(np.all(pred == truth))
    pixel_acc = (pred == truth).sum() / truth.size
    return match, pixel_acc

def load_data(base_dir, max_data=300):
    filenames = os.listdir(base_dir)
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]
    rng = random.Random(42)
    rng.shuffle(data_files)
    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            task_examples = json.load(fp)
        _, eval_examples = split_examples(task_examples)
        if len(eval_examples) < 4:
            continue
        rng.shuffle(eval_examples)
        sample = eval_examples[:4]
        train_sample = sample[:3]
        test_sample = sample[3]
        dataset.append({
            'task': os.path.basename(fn).split(".")[0],
            'train': train_sample,
            'test': [{'input': test_sample['input'], 'output': test_sample['output']}],
            'test_input': [{'input': test_sample['input']}],
            'test_output': [test_sample['output']],
        })
        if len(dataset) >= max_data:
            break
    df = pd.DataFrame(dataset)
    return df

def evaluate_checkpoint(checkpoint_dir, dataset, solver_token, ckpt_idx, total_ckpt):
    solver = ARCSolver(token=solver_token)
    solver.prepare_evaluation(checkpoint_dir=checkpoint_dir)
    
    match_count = 0
    pixel_total = 0
    pixel_correct = 0
    task_lines = []

    for idx, data in enumerate(tqdm(dataset, desc=f"[{ckpt_idx+1}/{total_ckpt}] {checkpoint_dir}")):
        pred = solver.predict(data["train"], data["test"][0]["input"])
        match, pixel_acc = check_match_and_pixel_accuracy(pred, data["test"][0]["output"])
        match_count += match
        pixel_total += np.array(data["test"][0]["output"]).size
        pixel_correct += pixel_acc * np.array(data["test"][0]["output"]).size
        task_lines.append(f"[Task {idx+1}] {data['task']}: match={match}, pixel_acc={pixel_acc*100:.2f}%\n")
        if VISUALIZE:
            plot_arc_example(data["train"], data["test"][0]["input"], pred, task_id=data["task"])
            input("Press Enter to continue to the next task...")


    whole_acc = (match_count / len(dataset)) * 100
    pixel_acc = (pixel_correct / pixel_total) * 100
    model_id = solver.model_id

    import gc
    import torch
    del solver
    torch.cuda.empty_cache()
    gc.collect()
    
    return whole_acc, pixel_acc, task_lines, model_id

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    return parser.parse_args()

def main():
    if AUTO_EVAL:
        args = parse_args()
        global CHECKPOINT_BASE_DIR, MAX_DATA, VISUALIZE
        CHECKPOINT_BASE_DIR = args.checkpoint_dir
        MAX_DATA = 100
        VISUALIZE = False

    print(f"[Evaluation Target]: {CHECKPOINT_BASE_DIR}")

    kst_now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    token = os.getenv("HF_TOKEN_MKK")
    set_seed(1234)

    print("🔍 Loading evaluation data...")
    eval_df = load_data("../dataset", max_data=MAX_DATA)
    eval_dataset = Dataset.from_pandas(eval_df)

    print("🔍 Evaluating all checkpoints...")
    subdirs = [d for d in os.listdir(CHECKPOINT_BASE_DIR) if d.startswith("checkpoint-")]
    subdirs.sort(key=lambda x: int(x.split("-")[-1]) if x != "checkpoint-final" else float("inf"))

    timestamp = kst_now.strftime("%m%d_%H%M")
    log_group = f"{timestamp}_{CHECKPOINT_BASE_DIR.split('/')[-1]}"  # ✅ 수정: 그룹 이름 지정
    log_dir = os.path.join("logs", log_group)  # ✅ 수정: logs/group_name/
    os.makedirs(log_dir, exist_ok=True)
    

    for idx, ckpt_name in enumerate(subdirs):
        ckpt_path = os.path.join(CHECKPOINT_BASE_DIR, ckpt_name)
        if not os.path.isdir(ckpt_path):
            continue
        
        eval_start = time.time()
        whole_acc, pixel_acc, task_lines, model_id = evaluate_checkpoint(ckpt_path, eval_dataset, token, idx, len(subdirs))
        eval_end = time.time()

        # ✅ 수정: hyperparams 로딩
        hyperparams_path = os.path.join(ckpt_path, "hyperparams.json")
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path) as f:
                hyperparams = json.load(f)
        else:
            hyperparams = {}

        train_duration_str = hyperparams.pop("train_duration", None)
        partial_duration_str = hyperparams.pop("train_duration_partial", None)
        step = hyperparams.pop("step", None)

        elapsed = eval_end-eval_start
        elapsed_str = str(datetime.timedelta(seconds=round(elapsed)))

        summary = [
            f"[Summary]\n",
            f"Whole-grid accuracy: {whole_acc:.2f} %\n",
            f"Pixel-level accuracy: {pixel_acc:.2f} %\n",
            f"[Model]: {model_id}\n",
            f"[Checkpoint]: {ckpt_path}\n",
            f"[# Evaluation Tasks]: {len(eval_dataset)}\n",
            f"[Evaluation elapsed time]: {elapsed_str}\n",
        ]

        summary.append("\n[Training Duration]\n")
        if train_duration_str:
            summary.append(f"Total training time ({step} steps):     {train_duration_str}\n")
        if partial_duration_str:
            summary.append(f"Until this checkpoint ({step} steps):   {partial_duration_str}\n")

        summary.append("\n[Hyperparameters Used]\n")
        summary += [f"{k}: {v}\n" for k, v in hyperparams.items()]
        summary.append("\n")

        # ✅ 수정: log 파일 이름 지정
        log_path = os.path.join(log_dir, f"{ckpt_name}.log")
        with open(log_path, "w") as f:
            f.writelines(summary + task_lines)

        print(f"✅ Saved log: {log_path}")

if __name__ == "__main__":
    main()
