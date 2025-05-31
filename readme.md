0. 최초 모델
model_id = "meta-llama/Llama-3.2-3B-Instruct" 사용
dataset_stream.jsonl을 만들어서, moving windows 방식으로 각 태스크 별로 MAX_WINDOWS_PER_TASK 씩 순서대로 학습 (window = datapoint = 3+1 examples)
Best submit score : 0

1. loss-only-with-output-grid
기존에는 loss 계산시에 tokenize된 prompt까지 같이 학습됨
labels = [-100] * len(prompt_tokens) + output_tokens 를 추가하여, prompt 부분은 masking(-100)하고, output_tokens만 loss 계산에 포함되도록 함
Best submit score : 5

2. task-cycle-with-epoch
기존에는 training 시 하나의 task 별로 40개 정도 window를 학습한 다음 그다음 task로 넘어가는 방식
결국 training이 전부 끝나야 모든 task를 학습할 수 있었고, 후반부에 앞에서 학습한 task를 까먹을 가능성이 높음
그래서 epoch를 도입해서 하나의 task 별로 4개 정도 window를 학습하면서 넘어가고, 그다음 epoch에 다시 처음 task로 돌아와서 다음 window를 이어서 학습하도록 함
Best submit score : 10

3. lama-3.1-8B-Instruct
모델변경 Llama-3.2-3B-Instruct -> lama-3.1-8B-Instruct
training 시간은 3배 증가했는데, 성능은 나아지지 않아서 다시 Llama-3.2-3B-Instruct로 회귀

