import copy
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import CNN, LSTM

folder = Path(__file__).resolve().parent
labels = ["Claude", "GPT", "Llama"]


def tokenize(text):
    text = text.lower()
    pattern = r"\b(?:chatgpt|gpt[- ]?\d[\w.-]*|claude(?:[- ]\d[\w.-]*)?|llama(?:[- ]\d[\w.-]*)?|openai|anthropic)\b"
    text = re.sub(pattern, "<model>", text)
    return re.findall(r"<model>|\w+|[^\w\s]", text)


def get_texts(data, mode):
    texts = []
    for row in data.itertuples():
        prompt = tokenize(row.LLM_Input)[:96]
        response = tokenize(row.LLM_output)[:192]
        if mode == "input":
            text = prompt
        elif mode == "output":
            text = response
        else:
            text = prompt + ["<sep>"] + response
        if len(text) == 0:
            text = ["<unk>"]
        texts.append(text)
    return texts


def make_loader(data, texts, vocab, max_length, shuffle):
    words = torch.zeros(len(texts), max_length, dtype=torch.long)
    lengths = []
    targets = []
    for index, text in enumerate(texts):
        lengths.append(len(text))
        for position, token in enumerate(text):
            words[index, position] = vocab.get(token, 1)
    for label in data["LLM_name"]:
        targets.append(labels.index(label))

    dataset = TensorDataset(words, torch.tensor(lengths), torch.tensor(targets))
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


def evaluate(model, loader):
    model.eval()
    true_labels = []
    predictions = []
    total_loss = 0
    with torch.no_grad():
        for words, lengths, targets in loader:
            scores = model(words, lengths)
            loss = nn.functional.cross_entropy(scores, targets, reduction="sum")
            total_loss += loss.item()
            true_labels.extend(targets.tolist())
            predictions.extend(scores.argmax(dim=1).tolist())
    return true_labels, predictions, total_loss / len(true_labels)


def run(data, model_name, mode):
    torch.manual_seed(42)
    train_data = data[data["split"] == "train"]
    val_data = data[data["split"] == "val"]
    test_data = data[data["split"] == "test"]
    train_texts = get_texts(train_data, mode)
    val_texts = get_texts(val_data, mode)
    test_texts = get_texts(test_data, mode)

    counts = Counter()
    for text in train_texts:
        counts.update(text)
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2, "<model>": 3}
    for token, count in counts.most_common(12000):
        if token not in vocab and len(vocab) < 12000:
            vocab[token] = len(vocab)

    if mode == "input":
        max_length = 96
    elif mode == "output":
        max_length = 192
    else:
        max_length = 289
    train_loader = make_loader(train_data, train_texts, vocab, max_length, True)
    val_loader = make_loader(val_data, val_texts, vocab, max_length, False)
    test_loader = make_loader(test_data, test_texts, vocab, max_length, False)

    if model_name == "CNN":
        model = CNN(len(vocab))
    else:
        model = LSTM(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, 13):
        model.train()
        for words, lengths, targets in train_loader:
            optimizer.zero_grad()
            scores = model(words, lengths)
            loss = loss_function(scores, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_labels, val_predictions, val_loss = evaluate(model, val_loader)
        print(model_name, mode, "epoch", epoch, "validation loss", round(val_loss, 4), flush=True)
        if val_loss < best_loss - 0.00001:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 3:
            break

    model.load_state_dict(best_weights)
    true_labels, predictions, test_loss = evaluate(model, test_loader)
    return {"model": model_name, "input": mode,
            "accuracy": accuracy_score(true_labels, predictions),
            "macro_f1": f1_score(true_labels, predictions, average="macro", zero_division=0),
            "best_epoch": best_epoch}


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = pd.read_csv(folder / "data.csv")
    results = []
    for model_name in ["CNN", "LSTM"]:
        for mode in ["input", "output", "both"]:
            results.append(run(data, model_name, mode))
    results = pd.DataFrame(results)
    results.to_csv(folder / "results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
