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