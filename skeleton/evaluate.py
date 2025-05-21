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

CHECKPOINT_DIR = "checkpoints/checkpoint-4"
MAX_DATA = 50   # 3+1 조합에 대한 task evaluation 횟수

def split_examples(task_examples, train_ratio=0.8, seed=42):
    """
    ARC task의 예시를 8:2로 분할
    """
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
    else:
        match = int(np.all(pred == truth))
        total_pixels = truth.size
        correct_pixels = (pred == truth).sum()
        pixel_acc = correct_pixels / total_pixels
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


def main():
    token = os.getenv("HF_TOKEN_MKK")
    solver = ARCSolver(token=token)
    solver.prepare_evaluation(checkpoint_dir=CHECKPOINT_DIR)

    set_seed(1234)

    eval_df = load_data("../dataset", max_data=MAX_DATA)
    eval_dataset = Dataset.from_pandas(eval_df)

    scores = []
    pixel_scores = []
    task_lines = []

    os.makedirs("logs", exist_ok=True)
    kst_now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    timestamp = kst_now.strftime("%m%d_%H%M")
    log_path = f"logs/{CHECKPOINT_DIR.split('/')[-1]}_{timestamp}.log"

    for idx, data in enumerate(tqdm(eval_dataset), start=1):
        task_name = data["task"]

        pred = solver.predict(data["train"], data["test"][0]["input"])
        match, pixel_acc = check_match_and_pixel_accuracy(pred, data["test"][0]["output"])

        scores.append(match)
        pixel_scores.append(pixel_acc)

        task_lines.append(f"[Task {idx}] {task_name}: match={match}, pixel_acc={pixel_acc*100:.2f}%\n")

    score = np.mean(scores) * 100
    pix_score = np.mean(pixel_scores) * 100
    elapsed = datetime.datetime.now(ZoneInfo("Asia/Seoul")) - kst_now

    hyperparams_path = os.path.join(CHECKPOINT_DIR, "hyperparams.json")
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
    else:
        hyperparams = {}

    summary = [
        f"[Summary]\n",
        f"Whole-grid accuracy: {score:.2f} %\n",
        f"Pixel-level accuracy: {pix_score:.2f} %\n",
        f"[Evaluation time]: {str(elapsed)}\n",
        f"[Model]: {solver.model.config._name_or_path}\n",
        f"[Checkpoint]: {CHECKPOINT_DIR}\n",
        f"[# Evaluation Tasks]: {len(eval_dataset)}\n",
        "\n[Hyperparameters Used]\n"
    ]

    summary += [f"{k}: {v}\n" for k, v in hyperparams.items()]
    summary += "\n"

    with open(log_path, "w") as f:
        f.writelines(summary + task_lines)

    print(f"✅ Evaluation complete. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
