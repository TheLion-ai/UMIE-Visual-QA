import numpy as np
from datasets import load_dataset, load_from_disk, Image, Dataset
from umie_datasets.config.dataset_config import all_datasets
import typer
from typing_extensions import Annotated
import click
from rich.progress import Progress
import asyncio
import json_repair
import json
from rich import print

from utils import find_interesting_examples
from visualization import visualize_segmentation_mask
from llm import generate_openai, extract_json_string, continue_openai
from prompts import report_prompt, vqa_prompt
from constants import source_dataset, target_dataset

app = typer.Typer()


def get_dataset_names():
    return [config.dataset_name for config in all_datasets if config.masks]

@click.group()
def cli():
    pass

@cli.command()
@click.option('--name', type=click.Choice(get_dataset_names()), help="The name of the dataset.", prompt="Select dataset")
@click.option('--max-examples', default=1, help="Maximum number of examples to process.")
def build(name, max_examples):
    dataset_config = [config for config in all_datasets if config.dataset_name == name][0]

    dataset = load_dataset(source_dataset, name, split="train")
    interesting_dataset = find_interesting_examples(dataset, max_examples)
    interesting_dataset.save_to_disk(f"data/interesting_{name}")


@cli.command()
@click.option('--name', type=click.Choice(get_dataset_names()), help="The name of the dataset.", prompt="Select dataset")
@click.option('--workers', default=5, help="Number of concurrent workers for processing.")
@click.option('--limit', default=None, help="Limit the number of examples to process.")
def process(name, workers, limit):
    dataset_config = [config for config in all_datasets if config.dataset_name == name][0]
    
    dataset = load_from_disk(f"data/interesting_{name}")
    if limit is not None:
        dataset = dataset.select(range(min(len(dataset), int(limit))))
    print(f"Processing {len(dataset)} examples...")

    processed_examples = []

    async def process_dataset(dataset, dataset_config):
        tasks = []
        sem = asyncio.Semaphore(workers)
        progress = Progress()
        task = progress.add_task("[cyan]Processing examples...", total=len(dataset))
        
        progress.start()
        try:
            for i in range(len(dataset)):
                tasks.append(asyncio.create_task(process_example(i, dataset, dataset_config, sem, progress, task)))
            return await asyncio.gather(*tasks)
        finally:
            progress.stop()


    async def process_example(i, dataset, dataset_config, sem, progress, task):
        async with sem:
            try:
                example = dataset[i]
                image = np.array(example["image"])
                mask = np.array(example["mask"])
                filepath = visualize_segmentation_mask(image, mask, dataset_config.masks)
                
                report, messages = await generate_openai(filepath, report_prompt)
                qa = await continue_openai(messages, vqa_prompt)
                qa = extract_json_string(qa)
                qa = json_repair.loads(qa)
                qa = json.dumps(qa)


                processed_example = {**example, "report": report, "visualized_image": filepath, "qa" :qa}
                progress.update(task, advance=1)
                return processed_example
            except Exception as e:
                print(f"[red]Error[/red] processing example {i}: {e}")
                

    processed_examples = asyncio.run(process_dataset(dataset, dataset_config))

    processed_dataset = Dataset.from_list(processed_examples)
    processed_dataset = processed_dataset.cast_column("visualized_image", Image())

    processed_dataset.push_to_hub(target_dataset, name)



if __name__ == "__main__":
    cli()
    # process()
