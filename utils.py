from collections import defaultdict
from datasets import Dataset
import numpy as np
import PIL
from tqdm import tqdm
from random import shuffle

def group_into_cases(dataset: Dataset) -> defaultdict:
    cases = defaultdict(list)
    for i, study_id in enumerate(dataset["study_id"]):
        cases[study_id].append(i)
    return cases

def count_annotations(mask: "PIL.Image.Image") -> int:
    return len(np.unique(np.array(mask))) - 1  # Subtract 1 to exclude the background


def find_interesting_examples(dataset: Dataset, max_examples: int = 10):
    cases = group_into_cases(dataset)
    interesting_examples = []
    for study_id, indices in tqdm(cases.items()):
        mask_counts = [(i, count_annotations(dataset[i]["mask"])) for i in indices]
        mask_counts.sort(key=lambda x: x[1], reverse=True)
        interesting_indices, _ = zip(*mask_counts)
        interesting_indices = list(interesting_indices)
        shuffle(interesting_examples)
        interesting_examples.extend(interesting_indices[:max_examples])
    
    return dataset.select(interesting_examples)

