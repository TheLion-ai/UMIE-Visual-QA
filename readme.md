# UMIE Visual QA Dataset Creator

UMIE Visual QA Dataset Creator is a tool for generating synthetic image annotations and visual question-answering (VQA) datasets based on UMIE (Universal Medical Image Encoder) datasets. This tool is designed to help create high-quality training data for finetuning vision language models in the medical imaging domain.

## Features

- Utilizes UMIE (Universal Medical Image Encoder) datasets as a foundation
- Extracts interesting examples from medical imaging datasets
- Generates synthetic image annotations and reports using AI
- Creates visual question-answering (VQA) data pairs
- Visualizes image segmentation masks
- Asynchronous processing with progress tracking

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/umie-visual-qa-dataset-creator.git
   cd umie-visual-qa-dataset-creator
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Configure your dataset sources in `constants.py`:
   ```python
   source_dataset = "lion-ai/umie_datasets"
   target_dataset = "your-target-dataset"
   ```
   Replace `your-target-dataset` with where you want to push the processed data.

## Usage

The tool provides two main commands: `build` and `process`.

### Building a Dataset

To build a dataset of interesting examples from a UMIE dataset:

```
poetry run python main.py build --name DATASET_NAME --max-examples NUMBER
```
- `--name`: The name of the UMIE dataset to use (required, will prompt if not provided)
- `--max-examples`: Maximum number of examples per secase to select (default: 1)

### Processing a Dataset
```
poetry run python main.py process --name DATASET_NAME --workers NUMBER --limit NUMBER
```
To process a built dataset and generate synthetic annotations and VQA data:
- `--name`: The name of the UMIE dataset to process (required, will prompt if not provided)
- `--workers`: Number of concurrent workers for processing (default: 5)
- `--limit`: Limit the number of examples to process. Can be usefull for testing purposes. Defaults to None which means processing the whole dataset
## Output

The processed dataset includes:
- Original medical images
- Segmentation mask visualizations
- AI-generated reports and annotations
- Synthetic visual question-answering pairs

This data can be used to finetune vision language models for medical imaging tasks.


## License

This code is licensed under the [MIT License](LICENSE). To use the datasets follow their respected liceses