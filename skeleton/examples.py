import os
import json
import numpy as np
from utils import render_grid
from rich import print

# ✅ 하이퍼파라미터
N_TASKS = 5              # 시각화할 태스크 개수
EXAMPLES_PER_TASK = 3    # 각 태스크 내 시각화할 input/output 예제 수

def load_arc_tasks(base_dir):
    filenames = sorted([f for f in os.listdir(base_dir) if f.endswith(".json")])
    rng = np.random.default_rng(42)
    selected_files = rng.choice(filenames, size=N_TASKS, replace=False)

    tasks = []
    for filename in selected_files:
        with open(os.path.join(base_dir, filename)) as f:
            examples = json.load(f)
            tasks.append((filename.split(".")[0], examples))
    return tasks

def visualize_tasks(tasks):
    for task_name, examples in tasks:
        print(f"\n[bold yellow]🧩 Task: {task_name}[/bold yellow] (total {len(examples)} examples)")

        num_to_show = min(EXAMPLES_PER_TASK, len(examples))
        for i in range(num_to_show):
            example = examples[i]
            print(f"[cyan]Example {i+1}[/cyan]")
            print("[green]Input:[/green]")
            render_grid(example['input'])
            print("[green]Output:[/green]")
            render_grid(example['output'])
            print("-" * 40)

def main():
    base_dir = "../dataset"
    tasks = load_arc_tasks(base_dir)
    visualize_tasks(tasks)

if __name__ == "__main__":
    main()
