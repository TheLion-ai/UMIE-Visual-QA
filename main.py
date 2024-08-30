import numpy as np
from datasets import load_dataset, load_from_disk
from umie_datasets.config.dataset_config import all_datasets
import typer
from typing_extensions import Annotated

from utils import find_interesting_examples
from visualization import visualize_segmentation_mask
from llm import generate_openai
from prompts import report_prompt

app = typer.Typer()


def get_dataset_names():
    return [config.dataset_name for config in all_datasets]


@app.command()
def build_dataset(
    name: Annotated[
        str,
        typer.Option(help="The name to say hi to.", autocompletion=get_dataset_names),
    ],
    max_examples: int=1,
):

    dataset_config = [
        config for config in all_datasets if config.dataset_name == name
    ][0]

    dataset = load_dataset("lion-ai/umie_datasets", name, split="train")
    interesting_dataset = find_interesting_examples(
        dataset, max_examples
    )
    interesting_dataset.save_to_disk(f"interesting_{name}")

@app.command()
def process(
    name: Annotated[
        str,
        typer.Option(help="The name to say hi to.", autocompletion=get_dataset_names),
    ],
    limit :int = 10
):
    dataset_config = [
        config for config in all_datasets if config.dataset_name == name
    ][0]
    
    dataset = load_from_disk(f"interesting_{name}")
    dataset = dataset.select(range(min(len(dataset), limit)))
    for i in range(len(dataset)):
        image = np.array(dataset[i]["image"])
        mask = np.array(dataset[i]["mask"])
        filepath = visualize_segmentation_mask(image, mask, dataset_config.masks)
        
        report = generate_openai(filepath, report_prompt)
        # Append report and visualized image to the example
        dataset[i]["report"] = report
        dataset[i]["visualized_image"] = filepath

        print(f"Processed example {i+1}/{len(dataset)}")

    # Save the updated dataset
    dataset.save_to_disk(f"processed_{name}")


if __name__ == "__main__":
    # app()
    process(name = "kits23")

# dataset
