# NLP Named-entity recognition (NER)

Deadline: `December 8 2024`

## Install dependencies
Python 3.11
>Note Python <= 3.11 is required

```bash
pip install -r requirements.txt
```

## Download dataset

In this work the [CoNLL003](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion/data) dataset is used.

```bash
mkdir -p data/
pushd data/
    kaggle datasets download -d alaakhaled/conll003-englishversion
    unzip conll003-englishversion.zip
    rm -rf conll003-englishversion.zip
popd
```
## Structure of project
