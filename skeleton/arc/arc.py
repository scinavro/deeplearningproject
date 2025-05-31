from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        print("ARCSolver Created")

        config_path = "artifacts/config/config.yml"
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='eager', # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16, # Set the data type for the model
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

        # pad_token이 정의되지 않았거나 eos_token과 같은 경우
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            print("⚠️ pad_token이 eos_token과 동일합니다. '<pad>' 토큰을 새로 등록합니다.")
            
            # '<pad>'가 없는 경우 추가
            if "<pad>" not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                print("✅ '<pad>' 토큰이 vocabulary에 추가되었습니다.")

            # 모델 임베딩 사이즈 재조정
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def format_prompt(self, datapoint):
        """
        Args:
            datapoint (dict): contains 'train' and 'test' keys

        Returns:
            dict with 'input_ids', 'labels', and raw 'input' and 'output' grids
        """
        training_data = datapoint['train']
        test_input = datapoint['test'][0]['input']
        test_output = datapoint['test'][0].get('output', None)  # evaluation 때는 'output' 없음

        tokenizer = self.tokenizer
        eos = tokenizer.eos_token or "<|endoftext|>"

        # 1. System + user 프롬프트
        sys = tokenizer.encode(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system_prompt,
            add_special_tokens=False
        )

        user = tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n", add_special_tokens=False)

        inp_desc = tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = tokenizer.encode("output:\n", add_special_tokens=False)

        for ex in training_data:
            inp = self.format_grid(ex['input'])
            out = self.format_grid(ex['output'])
            user += inp_desc + inp + out_desc + out

        user += tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)
        user += inp_desc + self.format_grid(test_input)
        user += tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)

        prompt_tokens = sys + user
        assistant_header = tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False)

        prompt_tokens += assistant_header

        
        if test_output is not None:
            output_tokens = self.format_grid(test_output)
            input_ids = prompt_tokens + output_tokens   # training: 정답 (output_tokens) 알려줌 -> loss 계산에 사용
            labels = [-100] * len(prompt_tokens) + output_tokens    # 정답 부분 제외하고 마스킹 (-100) -> loss 계산 시 정답 부분만 고려
        else:
            input_ids = prompt_tokens   # evaluation: 모델은 정답 모름 → generate할 것
            labels = None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input": test_input,
            "output": test_output,
            "train": training_data,
        }

    def predict(self, examples, questions_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "train": examples,
            "test": [{"input": questions_input}]
        }

        prompt = self.format_prompt(datapoint)  # test output은 포함 X
        input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150,
        )

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=config,
            ).squeeze(0).cpu()

        # 👉 명시적으로 prompt 길이만큼 자르기
        prompt_len = input_ids.shape[1]
        gen_tokens = output[prompt_len:].tolist()

        test_input = np.array(prompt['input'])

        try:
            grid_list = self.parse_grid(gen_tokens)
            # Check for row consistency
            if not grid_list or not all(len(row) == len(grid_list[0]) for row in grid_list):
                raise ValueError("Inconsistent row lengths in grid_list")
            grid = np.array(grid_list)
            if grid.ndim != 2:
                raise ValueError("Parsed grid is not 2D")

            # shape 보정 (예전 방식 유지)
            train_input = np.array(prompt['train'][0]['input'])
            train_output = np.array(prompt['train'][0]['output'])

            if train_input.shape == train_output.shape:
                x, y = test_input.shape
            else:
                x = (train_output.shape[0] * test_input.shape[0]) // train_input.shape[0]
                y = (train_output.shape[1] * test_input.shape[1]) // train_input.shape[1]

            grid = grid[:x, :y]

        except Exception as e:
            print(f"[Warning] parse_grid failed: {e}")
            grid = np.random.randint(0, 10, (test_input.shape[0], test_input.shape[1]))

        return grid

    def prepare_evaluation(self, checkpoint_dir="artifacts/checkpoint-final"):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_dir)
        self.model.eval()

    def __del__(self):
        print("ARCSolver destroyed")
