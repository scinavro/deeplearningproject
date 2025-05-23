import os
import json
import random
import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from arc import ARCSolver
from transformers import set_seed
from datasets import Dataset
from zoneinfo import ZoneInfo

MAX_DATA = 100
CHECKPOINT_BASE_DIR = "checkpoints/loss-only-with-output-grid-2"

def split_examples(task_examples, train_ratio=0.9, seed=42):
    rnd = random.Random(seed)
    task_examples = task_examples.copy()
    rnd.shuffle(task_examples)
    split = int(len(task_examples) * train_ratio)
    return task_examples[:split], task_examples[split:]

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

    whole_acc = (match_count / len(dataset)) * 100
    pixel_acc = (pixel_correct / pixel_total) * 100
    return whole_acc, pixel_acc, task_lines

def find_best_checkpoint(base_dir, dataset, token):
    candidates = []
    subdirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") or d == "checkpoint-final"]
    subdirs.sort(key=lambda x: (x != "checkpoint-final", int(x.split("-")[-1]) if x != "checkpoint-final" else float("inf")))

    for idx, name in enumerate(subdirs):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            whole_acc, pixel_acc, task_lines = evaluate_checkpoint(path, dataset, token, idx, len(subdirs))
            print(f"{name}: whole={whole_acc:.2f}%, pixel={pixel_acc:.2f}%")
            candidates.append((pixel_acc, name, path, whole_acc, task_lines))

    if not candidates:
        raise ValueError("❌ No valid checkpoints found in directory.")

    best = max(candidates)  # 최대 pixel accuracy 기준
    print(f"✅ Best checkpoint: {best[1]} (pixel={best[0]:.2f}%)")
    return best  # (pixel_acc, name, path, whole_acc, task_lines)

def main():
    kst_now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))

    token = os.getenv("HF_TOKEN_MKK")
    set_seed(1234)

    print("🔍 Loading evaluation data...")
    eval_df = load_data("../dataset", max_data=MAX_DATA)
    eval_dataset = Dataset.from_pandas(eval_df)

    print("🔍 Searching best checkpoint...")
    best_pixel_acc, best_ckpt_name, best_ckpt_path, best_whole_acc, best_task_lines = find_best_checkpoint(CHECKPOINT_BASE_DIR, eval_dataset, token)

    hyperparams_path = os.path.join(best_ckpt_path, "hyperparams.json")
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
    else:
        hyperparams = {}

    elapsed = datetime.datetime.now(ZoneInfo("Asia/Seoul")) - kst_now

    summary = [
        f"[Summary]\n",
        f"Whole-grid accuracy: {best_whole_acc:.2f} %\n",
        f"Pixel-level accuracy: {best_pixel_acc:.2f} %\n",
        f"[Model]: {ARCSolver(token).model.config._name_or_path}\n",
        f"[Checkpoint]: {best_ckpt_path}\n",
        f"[# Evaluation Tasks]: {len(eval_dataset)}\n",
        f"[Evaluation time]: {str(elapsed)}\n",
        "\n[Hyperparameters Used]\n"
    ]
    summary += [f"{k}: {v}\n" for k, v in hyperparams.items()]
    summary += ["\n"]

    timestamp = kst_now.strftime("%m%d_%H%M")
    os.makedirs("logs", exist_ok=True)
    log_suffix = best_ckpt_name.replace("checkpoint-", f"{CHECKPOINT_BASE_DIR.split('/')[-1]}-")
    log_path = f"logs/{timestamp}_{log_suffix}.log"

    with open(log_path, "w") as f:
        f.writelines(summary + best_task_lines)

    print(f"✅ Evaluation complete. Log saved to: {log_path}")

if __name__ == "__main__":
    main()
