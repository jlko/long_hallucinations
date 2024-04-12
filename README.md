# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains the code necessary to reproduce the paragraph-length experiments of the Nature submission 'Detecting Hallucinations in Large Language Models Using Semantic Entropy'.


## System Requirements

We here discuss hardware and software system requirements.

### Hardware Dependencies

Generally speaking, our experiments require modern computer hardware which is suited for machine learning research.

Requirements regarding the system's CPU and RAM size are relatively modest: any reasonably modern system should suffice, e.g. a system with an Intel 10th generation CPU and 16 GB of system memory or better.

More importantly, all our experiments make use of one or more Graphics Processor Units (GPUs) to speed up LLM inference.
Without a GPU, it is not feasible to reproduce our results in a reasonable amount of time.


### Software Dependencies

Our code relies on Python 3.11 with PyTorch 2.1.

Our systems run the Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-89-generic x86_64) operating system.

In [environment_export.yaml](environment_export.yaml) we list the exact versions for all Python packages.

Although we have not tested this, we would expect our code to be compatible with other operating systems, Python versions, and versions of the Python libraries that we use.


## Installation Guide


To install Python with all necessary dependencies, we recommend the use of conda, and we refer to [https://conda.io/](https://conda.io/) for an installation guide.


After installing conda, you can set up and activate a new conda environment with all required packages by executing the following commands from the root folder of this repository in a shell:


```
conda-env update -f environment.yaml
conda activate semantic_uncertainty
```

The installation should take around 15 minutes.

Our experiments rely on [Weights & Biases (wandb)](https://wandb.ai/) to log and save individual runs.
While wandb will be installed automatically with the above conda script, you may need to log in with your wandb API key upon initial execution.


Our experiments with sentence-length generation use GPT models from the OpenAI API.
Please set the environment variable `OPENAI_API_KEY` to your OpenAI API key in order to use these models.
Note that OpenAI charges a cost per input token and per generated token.
Costs for reproducing our results vary depending on experiment configuration, but, without any guarantee, should be around 20 dollars to reproduce our results.


The FactualBio dataset is included with this codebase in the file `data.py`.


## Reproduction Instructions


Run

```
python hallucination.py --model=QADebertaEntailment
python hallucination.py --model=SelfCheckBaseline
python hallucination.py --model=PTrueOriginalBaseline
```

to reproduce results for paraphrase-length generations.

The expected runtime of running all scripts in parallel is about 2 hours.

To evaluate the run and obtain a barplot similar to those of the paper, open the the Jupyter notebook in [notebooks/example_evaluation.ipynb](notebooks/example_evaluation.ipynb), populate the wandb ids in the second cell with the ids assigned to your runs, and execute all cells of the notebook.


We refer to [https://jupyter.org/](https://jupyter.org/) for more information on how to start the Jupter notebook server.



## Repository Structure

We here give a brief overview over the various components of the code.

* `hallucination.py`: Main script that iterates over datapoints of FactualBio dataset.
* `data.py`: Contains FactualBio dataset.
* `models.py`: Implements different hallucination detection methods.
