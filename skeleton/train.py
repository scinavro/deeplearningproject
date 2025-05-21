# train_streaming.py

import os
import json
import torch
from tqdm import tqdm
from arc import ARCSolver
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from transformers import get_scheduler
from torch.optim import AdamW

from utils import split_examples

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4

CHECKPOINT_NAME = "checkpoint-5"
MAX_WINDOWS_PER_TASK = 100 # 각 task에서 학습할 조합(3+1 in/out)의 최대 개수
MAX_STEPS = 27000 # 최대 300*0.9*max_windows_per_task

def save_hyperparameters(checkpoint_dir):
    import json

    hyperparams = {
        "LORA_RANK": LORA_RANK,
        "LORA_ALPHA": LORA_ALPHA,
        "LORA_DROPOUT": LORA_DROPOUT,
        "LEARNING_RATE": LEARNING_RATE,
        "CHECKPOINT_NAME": CHECKPOINT_NAME,
        "MAX_WINDOWS_PER_TASK": MAX_WINDOWS_PER_TASK,
        "MAX_STEPS": MAX_STEPS,
    }

    with open(os.path.join(checkpoint_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

def merge_json_files(dataset_dir, output_file):
    """
    여러 ARC JSON 파일을 하나의 .jsonl 스트림 파일로 병합
    """
    with open(output_file, "w") as out_fp:
        for fn in os.listdir(dataset_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(dataset_dir, fn)) as f:
                task_data = json.load(f)
            if len(task_data) < 2:
                continue
            out_fp.write(json.dumps({"task": fn, "examples": task_data}) + "\n")

def main():
    dataset_dir = "../dataset"
    merged_path = "./dataset_stream.jsonl"
    checkpoint_dir = f"checkpoints/{CHECKPOINT_NAME}"
    final_checkpoint_dir = checkpoint_dir + "-final"
    token = os.getenv("HF_TOKEN_MKK")

    # Step 1: merge all json into .jsonl
    if not os.path.exists(merged_path):
        print("Merging JSON files into jsonl format...")
        merge_json_files(dataset_dir, merged_path)

    # Step 2: load streaming dataset
    print("Loading streaming dataset...")
    stream_dataset = load_dataset(
        "json", 
        data_files=merged_path, 
        split="train", 
        streaming=True
    )

    # Step 3: Initialize solver and prepare LoRA model
    solver = ARCSolver(token=token)
    model = solver.model
    tokenizer = solver.tokenizer

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Step 4: streaming + sliding window training loop
    print("Starting training loop...")

    step = 0
    window_size = 3 # 추론 전 3개 input-output 조합 미리보기
    # best_loss = float("inf")
    early_stop = False

    for task in tqdm(stream_dataset):
        examples = task["examples"]
        if len(examples) < window_size + 1:
            continue

        train_examples, _ = split_examples(examples)  # eval 부분은 무시
        max_start = len(train_examples) - window_size
        
        windows_processed = 0
        # task_loss_sum = 0.0
        # task_loss_count = 0
        

        for start in range(0, max_start, window_size + 1):
            if windows_processed >= MAX_WINDOWS_PER_TASK:
                break

            train_chunk = train_examples[start:start + window_size]
            test_idx = start + window_size
            if test_idx >= len(train_examples):
                break
            test_example = train_examples[test_idx]

            datapoint = {
                "train": train_chunk,
                "test": [test_example]
            }

            prompt = solver.format_prompt(datapoint)
            input_ids = torch.tensor(prompt["input_ids"], dtype=torch.long).unsqueeze(0).to(solver.device)
            labels = input_ids.clone()
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # task_loss_sum += loss.item()
                # task_loss_count += 1
                step += 1

                if step % MAX_WINDOWS_PER_TASK == 0:
                    print(f"[Step {step}] loss: {loss.item():.4f}")

                if step >= MAX_STEPS:
                    early_stop = True
                    break
                
                windows_processed += 1

            except torch.cuda.OutOfMemoryError:
                print(f"[Step {step}] ⚠️ CUDA OOM: skipping example")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
        
        if early_stop:
            break

        # if task_loss_count > 0:
        #     avg_task_loss = task_loss_sum / task_loss_count
        #     if avg_task_loss < best_loss:
        #         best_loss = avg_task_loss
        #         print(f"🔥[Step {step}] New best avg loss: {best_loss:.4f} (task: {task['task']})")
        #         print( f"saving model to {checkpoint_dir}")
        #         model.save_pretrained(checkpoint_dir)
        #         tokenizer.save_pretrained(checkpoint_dir)
        #         save_hyperparameters(checkpoint_dir)

    print(f"Saving final model to {final_checkpoint_dir}")
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    save_hyperparameters(final_checkpoint_dir)

if __name__ == "__main__":
    main()
