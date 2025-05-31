import os
import json

dataset_dir = "../dataset"
task_example_counts = {}

for filename in os.listdir(dataset_dir):
    if filename.endswith(".json"):
        path = os.path.join(dataset_dir, filename)
        with open(path, "r") as f:
            examples = json.load(f)
        task_name = filename.replace(".json", "")
        task_example_counts[task_name] = len(examples)

# 출력
for task, count in sorted(task_example_counts.items()):
    print(f"{task}: {count} examples")
