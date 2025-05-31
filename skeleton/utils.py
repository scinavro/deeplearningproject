from pathlib import Path

import numpy as np

from rich.console import Console
from rich.text import Text
from typing import List

color_map = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "white",
    8: "bright_red",
    9: "bright_green",
}

console = Console()

def make_rich_lines(grid: List[List[int]]) -> List[Text]:
    lines = []
    for row in grid:
        visual = Text()
        for cell in row:
            color = color_map.get(cell, "white")
            visual.append("  ", style=f"on {color}")
        raw = Text("  " + str(row))
        visual.append(raw)
        lines.append(visual)
    return lines

def render_grid(grid: List[List[int]]):
    lines = make_rich_lines(grid)
    for line in lines:
        console.print(line)

def get_base_model(model_name):
    available_models = [
        "meta-llama/Llama-3.2-1B",
    ]
    assert model_name in available_models, f"{model_name} is not available."

import random

def split_examples(task_examples, train_ratio=0.9, seed=42):
    """
    하나의 ARC task 예시 리스트를 train/eval 예시로 분리
    - task_examples: List[dict] with keys 'input', 'output'
    - 반환: (train_list, eval_list)
    """
    random.Random(seed).shuffle(task_examples)  # 동일 분할 유지 위해 seed 고정
    split_idx = int(len(task_examples) * train_ratio)
    return task_examples[:split_idx], task_examples[split_idx:]

import sys
import os

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)  # 에러도 함께 기록

import matplotlib.pyplot as plt
import numpy as np

def plot_arc_example(train_examples, test_input, generated_output, task_id=None):
    """
    Visualize 3 training pairs + test input/output
    """
    fig, axes = plt.subplots(4, 2, figsize=(6, 12))  # 4 rows: 3 train, 1 test

    for i, ex in enumerate(train_examples):
        axes[i, 0].imshow(np.array(ex['input']), cmap='tab20', vmin=0, vmax=9)
        axes[i, 0].set_title(f"Train {i+1} Input")
        axes[i, 1].imshow(np.array(ex['output']), cmap='tab20', vmin=0, vmax=9)
        axes[i, 1].set_title(f"Train {i+1} Output")

    axes[3, 0].imshow(np.array(test_input), cmap='tab20', vmin=0, vmax=9)
    axes[3, 0].set_title("Test Input")

    axes[3, 1].imshow(np.array(generated_output), cmap='tab20', vmin=0, vmax=9)
    axes[3, 1].set_title("Generated Output")

    for ax in axes.flatten():
        ax.axis('off')

    if task_id:
        fig.suptitle(f"Task {task_id}", fontsize=14)

    plt.tight_layout()
    plt.show()