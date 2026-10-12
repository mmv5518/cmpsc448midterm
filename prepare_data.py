import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
revision = "cd543a149df89434d8a54582c0151c0b945c3d20"
models = {"GPT": "gpt4_0613", "Claude": "claude-2", "Llama": "llama-2-7b-chat-hf"}


def main():
    responses = {}
    for family, model in models.items():
        url = f"https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/{revision}/results/{model}/model_outputs.json"
        with urllib.request.urlopen(url, timeout=90) as response:
            records = json.loads(response.read())

        responses[family] = {}
        for record in records:
            if record["generator"] != model:
                raise ValueError("Unexpected model label")
            prompt = re.sub(r"\s+", " ", record["instruction"]).strip()
            if prompt and record["output"].strip():
                if prompt in responses[family]:
                    raise ValueError("Duplicate prompt in a source file")
                responses[family][prompt] = record["output"]
    shared = set(responses["GPT"])
    for family in models:
        shared = shared.intersection(responses[family])
    prompts = sorted(shared)
    if len(prompts) == 0:
        raise ValueError("No shared prompts found")

    order = np.random.default_rng(42).permutation(len(prompts))
    train_end = int(0.70 * len(prompts))
    val_end = train_end + int(0.15 * len(prompts))
    splits = {}
    for position, index in enumerate(order):
        if position < train_end:
            splits[index] = "train"
        elif position < val_end:
            splits[index] = "val"
        else:
            splits[index] = "test"

    rows = []
    for index, prompt in enumerate(prompts):
        for family in models:
            rows.append({"LLM_name": family, "LLM_Input": prompt,
                         "LLM_output": responses[family][prompt],
                         "split": splits[index]})

    data = pd.DataFrame(rows)
    data.to_csv(folder / "data.csv", index=False)
    print("Saved", len(data), "responses to data.csv")
    print(data.groupby(["split", "LLM_name"]).size())


if __name__ == "__main__":
    main()
