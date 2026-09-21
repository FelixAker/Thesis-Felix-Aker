"""
Fine-tune a trained student on one GLUE/SuperGLUE task and report the best
validation score (Section 4.6 of the thesis).

Training splits are capped at 10,000 examples and validation splits at 2,000,
so the fine-tuning cost is comparable across tasks. The reported metric is
accuracy for BoolQ, MultiRC and QQP, and F1 for MRPC.
"""
import argparse
import datasets
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functools import partial
from tqdm import tqdm


GLUE_SUBSETS = ['boolq', 'mrpc', 'multirc', 'qqp']
INPUTS = {
    'boolq': ['question', 'passage'],
    'mrpc': ['sentence1', 'sentence2'],
    'multirc': ['paragraph', 'question', 'answer'],
    'qqp': ['question1', 'question2'],
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--subset', type=str, required=True, choices=GLUE_SUBSETS)
parser.add_argument('--model_type', type=str, required=True, choices=['encoder', 'decoder'])
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--tokenizer_path', type=str, default="",
                    help="Tokenizer to use. Defaults to openai-community/gpt2 for decoder.")
parser.add_argument('--bs', '--batch_size', type=int, default=64)
parser.add_argument('--lr', '--learning_rate', type=float, default=5e-5)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)

def tokenize(examples, tokenizer, subset, truncate=True):
    batch = {
        "input_ids": [],
        "labels": [],
    }

    for i in range(len(examples['label'])):
        input_txt = " ".join([examples[txt][i] for txt in INPUTS[subset]])
        input_ids = tokenizer.encode(input_txt, truncation=truncate)
        batch["input_ids"].append(input_ids)
        batch["labels"].append([examples['label'][i]])

    return batch

def padding_collate_fn(batch, max_len=1024, left_padding=False):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            if left_padding:
                padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])
            else:
                padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch


def main():
    args = parser.parse_args()
    model_name = "prajjwal1/bert-tiny" if args.model_type == "encoder" else "sshleifer/tiny-gpt2"
    if not args.model_path:
        args.model_path = model_name

    if args.tokenizer_path:
        tokenizer_source = args.tokenizer_path
    else:
        tokenizer_source = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, subset=args.subset) # need to make the function unary for map()

    if args.subset in ['boolq', 'multirc']:
        dataset = datasets.load_dataset('super_glue', args.subset)
    else:
        dataset = datasets.load_dataset('glue', args.subset)

    # filter down to at most 10k training and 2k validation samples
    dataset['train'] = dataset['train'].select(range(min(10000, len(dataset['train']))))
    dataset['validation'] = dataset['validation'].select(range(min(2000, len(dataset['validation']))))
    if 'test' in dataset:
        del dataset['test']

    dataset = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=dataset['train'].column_names) 

    # config = AutoConfig.from_pretrained(args.model_path, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2, torch_dtype=torch.float32
    ).to(DEVICE)
    model.config.pad_token_id = 0

    train(model, dataset, args)


def train(model, dataset, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.bs, shuffle=True, collate_fn=padding_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(dataset['validation'], batch_size=args.bs, shuffle=False, collate_fn=padding_collate_fn)

    best = 0.0
    patience = args.patience
    for epoch in range(args.max_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(device=DEVICE)
            labels = batch['labels'].to(device=DEVICE)
            outputs = model(inputs, labels=labels, attention_mask=inputs != 0)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        result = evaluate(model, valid_dataloader, args.subset)
        print(f"Epoch: {epoch}, Result: {result}")

        if result > best:
            best = result
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break

    print(f"Best result: {best}")


def evaluate(model, dataloader, subset):
    correct = 0.0
    total = 0.0

    tp, fp, fn = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device=DEVICE)
            labels = batch['labels'][:, 0].to(device=DEVICE)

            outputs = model(input_ids, attention_mask=input_ids != 0).logits.argmax(-1)

            tp += ((outputs == 1) & (labels == 1)).sum().item()
            fp += ((outputs == 1) & (labels == 0)).sum().item()
            fn += ((outputs == 0) & (labels == 1)).sum().item()
            correct += (outputs == labels).sum().item()
            total += len(labels)

    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    if subset == 'mrpc':
        return f1
    else:
        return correct / total



if __name__ == '__main__':
    main()