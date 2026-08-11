# Hagiographics-relation-extraction

This repository contains code and data for training and evaluating relation extraction models on hagiographic texts. 
The main focus is on extracting relationships between entities in hagiographic narratives, which can be useful for historical and literary network analysis. The sources are writen in Latin and stem from the long 10th century.

For the annotation and validation of the data, we used Label Studio. This repository includes a notebook called 'import-export' that facilitates importing and exporting files to and from Label Studio.


---

## repository structure
The repository is organized as follows:

(1) folders:
- 'sample_texts': contains examples of hagiographic texts used for the relation extraction tasks.
- 'pre-annotations': contains an example LLM-prompt for doing a pre-annotations; the created pre-annotations are also stored in this folder.

(2) notebooks: 

Each step in this relation-extraction pipeline is documented in a separate notebook.
1. 'import-export': an all-purpose notebook for converting files from one format to another (e.g. Label Studio, GLiNER or SpaCy conversions)
2. 'GLiNER2-NER': a notebook to run the GLiNER2 model for Named Entity Recognition and to normalize and convert the NER file for subsequent usage in the rule-based relation extraction
3. 'LLM-pre-annotations': a step-by-step guide to creating pre-annotations using an LLM
4. 'evaluation': a notebook that compares the Ground-Truth annotations with the predictions of the relation extraction model and calculates evaluation metrics such as precision, recall, and F1-score and makes a table and confusion matrix visualisation.
5. 'GLiNER2-REX': a notebook that provides code for zero-shot relation extraction and LoRA-training using GLiNER2. Model fine-tuning has not been worked out.
6. 'rule-based-REX': a notebook for rule-based relation extraction using a SpaCy pipeline. 

(3) datasets:

To give you the possibility to test these notebooks without having to create your own dataset, we provide a small sample dataset in the 'sample_texts' folder. This dataset contains a few annotated hagiographic texts that can be used for testing and evaluation purposes.
1. 'rule-based-example.txt': a small sample text of a hagiographic narrative without any annotations. This text can be used to test the rule-based relation extraction and the GLiNER2 NER extraction that is needed before starting this notebook.
2. 'relation-annotated-example.json': a small sample text of a hagiographic narrative with annotated relations. This text can be used for the LoRA training and for every relation extraction workflow in this repository that you want to evaluate using ground truth annotations. The annotations are in the Label Studio format and can be converted to other formats using the 'import-export' notebook.
    
The 'relation-annotated-example.json' file contains 65 relations from 10 different relation types:

|relation type| number of relations |
|-------------|---------------------|
| is_owned_by     | 12                  |
| goes_to          | 11                  |
| is_located_at        | 10                  |
| resides_at           | 8                   |
| is_related_to        | 7                   |
| acts_on              | 6                   |
| is_called            | 4                   |
| owns                 | 3                   |
| founds               | 2                   |
| comes_from           | 2                   |
|TOTAL                 | 65                  |