from typing import List, Dict, Any
import re
import evaluate
from transformers import BertTokenizer, RobertaTokenizer

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute exact match ratio allowing for case and punctuation differences."""
    def normalize(text: str) -> str:
        """Normalize text by lowercasing and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w]', '', text)
        return text

    exact_match = 0
    for pred, ref in zip(predictions, references):
        if normalize(pred) == normalize(ref):
            exact_match += 1
    return exact_match / len(predictions)


def compute_bleu2(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references, max_order=2)

def compute_bleu4(predictions: List[str], references: List[str]) -> Dict[str, Any]: 
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, Any]: 
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)

roberta_tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")
bert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1")

def compute_bert_score(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, Any]]: 
    """Compute BERT score on roberta-large and biobert-large respectively."""
    results: Dict[str, Dict[str, Any]] = {}

    retokenized_predictions = roberta_tokenizer(
        predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_predictions = roberta_tokenizer.batch_decode(retokenized_predictions, skip_special_tokens=True)
    retokenized_labels = roberta_tokenizer(
        references, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_labels = roberta_tokenizer.batch_decode(retokenized_labels, skip_special_tokens=True)

    bert = evaluate.load("bertscore")
    roberta_results = bert.compute(predictions=truncated_predictions, references=truncated_labels, lang="en")
    results["roberta-large"] = {
        "precision": sum(roberta_results["precision"]) / len(roberta_results["precision"]), 
        "recall": sum(roberta_results["recall"]) / len(roberta_results["recall"]), 
        "f1": sum(roberta_results["f1"]) / len(roberta_results["f1"])
    }

    # truncate sentences to fit max_position_embeddings=512 of biobert
    retokenized_predictions = bert_tokenizer(
        predictions, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_predictions = bert_tokenizer.batch_decode(retokenized_predictions, skip_special_tokens=True)
    retokenized_labels = bert_tokenizer(
        references, padding="max_length", truncation=True, max_length=495, return_tensors="pt"
    )["input_ids"]
    truncated_labels = bert_tokenizer.batch_decode(retokenized_labels, skip_special_tokens=True)

    biobert_results = bert.compute(
        predictions=truncated_predictions,
        references=truncated_labels,
        model_type="dmis-lab/biobert-large-cased-v1.1",
        num_layers=24,
    )
    results["biobert-large"] = {
        "precision": sum(biobert_results["precision"]) / len(biobert_results["precision"]), 
        "recall": sum(biobert_results["recall"]) / len(biobert_results["recall"]), 
        "f1": sum(biobert_results["f1"]) / len(biobert_results["f1"])
    }

    return results


def compute_metrics(predictions: List[str], references: List[str], args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute BLEU, ROUGE, BERT scores and exact match ratio on given texts."""
    gathered_results: Dict[str, Dict[str, Any]] = {}

    if args["evaluate_exact_match"]:
        exact_match = compute_exact_match(predictions=predictions, references=references)
        gathered_results["exact_match"] = exact_match
        if args["verbose"]:
            print(f"EXACT match ratio: {exact_match}")

    if args["evaluate_bleu"]:
        bleu_results = compute_bleu2(predictions=predictions, references=references)
        gathered_results["bleu2"] = bleu_results
        if args["verbose"]:
            print(f"BLEU-2 score: {bleu_results}")
        bleu_results = compute_bleu4(predictions=predictions, references=references)
        gathered_results["bleu4"] = bleu_results
        if args["verbose"]:
            print(f"BLEU-4 score: {bleu_results}")

    if args["evaluate_rouge"]:
        rouge_results = compute_rouge(predictions=predictions, references=references)
        gathered_results["rouge"] = rouge_results
        if args["verbose"]:
            print(f"ROUGE score: {rouge_results}")
    
    if args["evaluate_bert_score"]:
        bert_results = compute_bert_score(predictions=predictions, references=references)
        gathered_results["bert"] = bert_results
        if args["verbose"]:
            for model_name, model_results in bert_results.items(): 
                print(f"BERT score with {model_name}: {model_results}")

    return gathered_results

import json
with open("data/prot2text/inference_results.json", "r") as f:
    data = json.load(f)

predictions = [item["predicted"] for item in data]
references = [item["label"] for item in data]

args = {
    "evaluate_exact_match": True,
    "evaluate_bleu": True,
    "evaluate_rouge": True,
    "evaluate_bert_score": True,
    "verbose": True
}

results = []
debug = 5000
metrics = compute_metrics(predictions=predictions[:debug], references=references[:debug], args=args)

with open("data/prot2text/evaluation_results_v2.json", "w") as f:
    json.dump(metrics, f, indent=4)