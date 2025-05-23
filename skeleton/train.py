# train.py

import os
import json
import torch
import time, datetime
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
CHECKPOINT_NAME = "task-cycle-with-epoch-2"
MAX_WINDOWS_PER_TASK = 4 # 각 task에서 학습할 조합(3+1 in/out)의 최대 개수
NUM_EPOCHS = 18
MAX_STEPS = 21600 # 원래 step은 300 * MAX_WINDOWS_PER_TASK * NUM_EPOCHS 까지 돌아야함. early_stop 하고 싶으면 그거보다 작게 설정하면 됨.
SAVE_EVERY_STEPS = 3600  # 중간 저장 간격

def save_hyperparameters(checkpoint_dir, train_duration=None, partial_duration=None, step=None):
    import json

    hyperparams = {
        "LORA_RANK": LORA_RANK,
        "LORA_ALPHA": LORA_ALPHA,
        "LORA_DROPOUT": LORA_DROPOUT,
        "LEARNING_RATE": LEARNING_RATE,
        "CHECKPOINT_NAME": CHECKPOINT_NAME,
        "MAX_WINDOWS_PER_TASK": MAX_WINDOWS_PER_TASK,
        "NUM_EPOCHS": NUM_EPOCHS,
        "MAX_STEPS": MAX_STEPS,
    }
    if train_duration:
        hyperparams["train_duration"] = train_duration
    if partial_duration:
        hyperparams["train_duration_partial"] = partial_duration
    if step:
        hyperparams["step"] = step

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
    checkpoint_dir_root = f"checkpoints/{CHECKPOINT_NAME}"
    checkpoint_final_dir = os.path.join(checkpoint_dir_root, "checkpoint-final")
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
    print("Initializing model...")
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
    train_start = time.time()

    step = 0
    window_size = 3 # 추론 전 3개 input-output 조합 미리보기
    early_stop = False

    for epoch in range(NUM_EPOCHS):
        for idx, task in enumerate(tqdm(stream_dataset)):
            examples = task["examples"]
            if len(examples) < window_size + 1:
                continue

            train_examples, _ = split_examples(examples)  # eval 부분은 무시
            max_start = len(train_examples) - window_size
            
            windows_processed = 0
            start_by_epoch = epoch*MAX_WINDOWS_PER_TASK*(window_size+1)

            for start in range(start_by_epoch, max_start, window_size + 1): # 다음 epoch에서는 저번에 안 봤던 example부터 window 설정
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
                labels = torch.tensor(prompt["labels"]).unsqueeze(0).to(solver.device)
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    step += 1
                    windows_processed += 1

                    # final_step % SAVE_EVERY_STEPS == 0 일때 마지막 모델의 중복 저장 방지
                    if step == 300 * MAX_WINDOWS_PER_TASK * NUM_EPOCHS:
                        break

                    # final_step은 아직 아닌데 MAX_STEPS까지만 하고 싶을 때
                    if step >= MAX_STEPS:
                        early_stop = True
                        break

                    # ✅ 중간 저장
                    if step % SAVE_EVERY_STEPS == 0:
                        partial_duration = time.time() - train_start
                        partial_duration_str = str(datetime.timedelta(seconds=round(partial_duration)))

                        step_ckpt_dir = os.path.join(checkpoint_dir_root, f"checkpoint-{step}")
                        print(f"Saving intermediate model to {step_ckpt_dir}")
                        model.save_pretrained(step_ckpt_dir)
                        tokenizer.save_pretrained(step_ckpt_dir)
                        save_hyperparameters(step_ckpt_dir, partial_duration=partial_duration_str, step=step)
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"[Step {step}] ⚠️ CUDA OOM: skipping example")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

            # print("start_by_epoch:", start_by_epoch)
            print(f"[Epoch {epoch+1} | Step {step} | Task {idx+1} | {task['task'].split('.')[0]}] loss: {loss.item():.4f}")
            
            if early_stop:
                print(f"early_stop by MAX_STEPS={MAX_STEPS}")
                break

    train_duration = time.time() - train_start
    train_duration_str = str(datetime.timedelta(seconds=round(train_duration)))

    print(f"Saving final model to {checkpoint_final_dir}")
    model.save_pretrained(checkpoint_final_dir)
    tokenizer.save_pretrained(checkpoint_final_dir)
    save_hyperparameters(checkpoint_final_dir, train_duration=train_duration_str, step=step)


if __name__ == "__main__":
    main()

    import subprocess
    print("🎯 Training completed. Now evaluating the model...")
    subprocess.run(["python", "evaluate.py"])
