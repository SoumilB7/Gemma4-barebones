---
language:
- en
- de
- fr
- zh
- pt
- nl
- ru
- ko
- it
- es
size_categories:
- 100K<n<1M
task_categories:
- conversational
dataset_info:
  features:
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: lang
    dtype: string
  - name: split
    dtype: string
  - name: dataset
    dtype: string
  - name: task
    dtype: string
  splits:
  - name: train
    num_bytes: 1568822476
    num_examples: 637495
  download_size: 730580350
  dataset_size: 1568822476
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset Card for TowerBlocks

TowerBlocks is the dataset used to train [TowerInstruct-v0.1](https://huggingface.co/Unbabel/TowerInstruct-7B-v0.1), a language model specialized for translation tasks such as machine translation (e.g. general, document, terminology-aware or context-aware translation), automatic post edition, named-entity recognition, gramatical error correction, and paraphrase generation.

- **Curated by:** Unbabel, Instituto Superior Técnico, CentraleSupélec, University of Paris-Saclay;
- **Language(s) (NLP):** English, Portuguese, Spanish, French, German, Dutch, Italian, Korean, Chinese, Russian;
- **License:** TowerBlocks contains data from many sources. We refer to the respective data sources below for information regarding licensing of the data.


## Dataset Details

TowerBlocks is a conversational dataset for translation related tasks created from a diverse set of high quality data sources:

| Data Source | Task(s) | 
| -------------- | ----------- | 
| [WMT14 to WMT21](https://www.statmt.org/wmt22/results.html) | General Translation |
| [WMT22](https://github.com/microsoft/gpt-MT) | Few-shot General Translation w/ Quality Shots |
| [NTREX](https://github.com/MicrosoftTranslator/NTREX) | General Translation |
| [Flores Dev](https://github.com/facebookresearch/flores) | General Translation |
| [FRMT](https://github.com/google-research/google-research/tree/master/frmt) | General Translation |
| [QT21](https://lindat.mff.cuni.cz/repository/xmlui/handle/11372/LRT-2390) | General Translation, Automatic Post Edition |
| [ApeQuest](https://apequest.wordpress.com/) | General Translation, Automatic Post Edition |
| [OPUS (Quality Filtered)](https://opus.nlpl.eu/) | General Translation |
| [MT-GenEval](https://github.com/amazon-science/machine-translation-gender-eval) | General Translation, Context-Aware Translation |
| [WMT20 to WMT22 Metrics MQM](https://www.statmt.org/wmt22/results.html) | Machine Translation Evaluation |
| [WMT17 to WMT22 Metrics Direct Assessments](https://www.statmt.org/wmt22/results.html) | Machine Translation Evaluation |
| [WMT21 Terminology Dev (filtered)](https://www.statmt.org/wmt21/terminology-task.html) | Terminology-aware Translation |
| [Tatoeba Dev (filtered)](https://github.com/Helsinki-NLP/Tatoeba-Challenge) | Multi-reference Translation |
| [MultiCoNER 2022 and 2023 Dev](https://registry.opendata.aws/multiconer/) | Named-entity Recognition | 
| [PAWS-X Dev](https://github.com/google-research-datasets/paws) | Paraphrase Generation |
| [UltraChat 200k (filtered)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Synthetic Chat data |
| [Glaive Code Assistant (filtered)](https://huggingface.co/datasets/glaiveai/glaive-code-assistant) | Code instructions |

The dataset was built by generating user instructions with records from each data source using a set of zero- and few-shot templates (with the exception of UltraChat 200k and Glaive Code Assistant which already contain user instructions).

### Dataset features

* `conversations` - The user and assistant dialog turns;
* `dataset` - Original dataset for the record;
* `lang` - Either the language or language pair of the original dataset;
* `task` - Task for the record (Can be used to identify the training templates for each task);
* `split` - Split of the original dataset from which the record was taken.

## Intended uses and limitations

TowerBlocks is intended for specializing language models towards translation related tasks via supervised finetuning.

## Citation

```bibtex
@misc{tower_llm_2024,
      title={Tower: An Open Multilingual Large Language Model for Translation-Related Tasks}, 
      author={Duarte M. Alves and José Pombal and Nuno M. Guerreiro and Pedro H. Martins and João Alves and Amin Farajian and Ben Peters and Ricardo Rei and Patrick Fernandes and Sweta Agrawal and Pierre Colombo and José G. C. de Souza and André F. T. Martins},
      year={2024},
      eprint={2402.17733},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```