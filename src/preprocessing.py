import pathlib
import random
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def set_seed(seed_value: int = 123) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value) 

def load_sentences(filepath: pathlib.Path) -> List[list]:
    """
    Loads sentences from a file for a NER task. Each sentence is represented by a list of tokens and their labels.
    :param filepath: The path to the data file
    :return: List of sentences, where each sentence is a list of tuples (token, label)
    """
    sentences = []  # current string
    final = []      # list of strings
    
    with open(filepath, 'r') as f:
        for line in f:
            # If the string is "-", sentence or document separator
            if line.strip() in {'-DOCSTART- -X- -X- O', ''}:
                if sentences: 
                    final.append(sentences)
                    sentences = []
            else:
                parts = line.strip().split(' ')
                # Make sure there are at least 4 parts:
                # token, part of speech, beginning of a named group, beginning of a named entity
                if len(parts) >= 4:  
                    token, tag = parts[0], parts[3]
                    sentences.append((token, tag))

    if sentences:
        final.append(sentences)
    
    return final

def analyze_data_quality(token_sentences: List[list]) -> dict:
    """
    Analyzes data quality for NER by checking:s
    1. Percentage of rows (tokens) with incorrect annotations.
    2. Presence of empty sentences.
    
    :param token_sentences: A list of sentences where each sentence is a list of tuples (token, label)
    :return: A dictionary with the results of the analysis
    """
    total_tokens, invalid_tokens, empty_sentences = 0,0,0

    for sentence in token_sentences:
        if not sentence:  
            empty_sentences += 1
            continue
        
        for token, tag in sentence:
            total_tokens += 1
            if not token or not tag:
                invalid_tokens += 1
    
    empty_sentence_percentage = (empty_sentences / len(token_sentences)) * 100 if token_sentences else 0
    invalid_token_percentage = (invalid_tokens / total_tokens) * 100 if total_tokens else 0
    
    return {
        "total_tokens": total_tokens,
        "invalid_tokens": invalid_tokens,
        "empty_sentences": empty_sentences,
        "empty_sentence_percentage": empty_sentence_percentage,
        "invalid_token_percentage": invalid_token_percentage
    }


def label_frequency_analysis(token_sentences: List[list]) -> dict:
    """
    Counts the number and percentage distribution of NER labels in the dataset.

    :param dataset: A list of sentences, where each sentence is a list of tuples (token, label)
    :return: A dictionary where keys are NER label types and values are a list of [number, percentage]
    """
    label_count = defaultdict(int)

    for sentence in token_sentences:
        for token, label in sentence:
            if label != 'O':
                _, ner_type = label.split('-')
                label_count[ner_type] += 1

    total_ner = sum(label_count.values())

    label_distribution = {}
    for ner_type, count in sorted(label_count.items()):  # Сортировка по алфавиту ключей
        percentage = round((count / total_ner) * 100, 2) if total_ner > 0 else 0
        label_distribution[ner_type] = percentage

    return label_distribution

def rare_and_dominant_categories(
        token_sentences: List[list],
        persentille: int = 25
    ) -> dict:
    """
    Identifies the top 25% and worst 25% of categories in the dataset based on the percentage distribution.
    
    :param dataset: A list of sentences, where each sentence is a list of tuples (token, label)
    :param top_percentage: The percentage of categories that are considered dominant (top 25%) or rare (worst 25%)
    :return: Dictionary with categories that are considered rare or dominant
    """
    total_tokens = 0
    label_count = defaultdict(int)
    
    for sentence in token_sentences:
        for _, label in sentence:
            total_tokens += 1
            label_count[label] += 1

    label_percentage = {
        label: (count / total_tokens) * 100 for label, count in label_count.items()
    }

    sorted_labels = sorted(label_percentage.items(), key=lambda x: x[1])

    top_count = int(len(sorted_labels) * (persentille / 100))
    top_categories = dict(sorted_labels[-top_count:])
    worst_categories = dict(sorted_labels[:top_count])

    result = {
        'total_tokens': total_tokens,
        'top_categories': top_categories,
        'worst_categories': worst_categories,
    }
    
    return result

def labels_distribution_charts(
        train_data: dict,
        test_data: dict,
        valid_data: dict
    ) -> None:
    """
    Draws three pie charts in one graph.

    :param data1: Data for the first diagram (list of values)
    :param data2: Data for the second diagram (list of values)
    :param data3: Data for the third diagram (list of values)
    :param labels: Labels for all diagrams (list of rows)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [train_data, test_data, valid_data]
    titles = ['Train dataset', 'Test dataset', 'Validation dataset']
    labels = list(train_data.keys())
    
    for i, ax in enumerate(axes):
        ax.pie(
            list(datasets[i].values()),
            labels=labels,
            autopct='%1.0f%%',
            pctdistance=1.1,
            labeldistance=1.3,
        )
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

def entity_neighbors(token_sentences: List[list]) -> dict:
    """
    Identifies which entity categories often follow each other.
    
    :param dataset: A list of sentences where each sentence is a list of tuples (token, label)
    :return: A dictionary with the frequency of entity neighborhoods
    """
    entity_pair_count = defaultdict(int)

    for sentence in token_sentences:
        previous_label = None 
        for token, label in sentence:
            if label != 'O': 
                if previous_label is not None:
                
                    entity_pair_count[(previous_label, label)] += 1
                previous_label = label
            else:
                previous_label = None

    return dict(entity_pair_count)

def plot_multiple_entity_neighbors(
        neighbor_train_data: dict,
        neighbor_test_data: dict,
        neighbor_valid_data: dict
    ) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    def plot_heatmap_on_ax(neighbor_stats, ax, title):
        labels = set()
        for pair in neighbor_stats.keys():
            labels.update(pair)
        labels = sorted(labels)
        matrix = {label: {l: 0 for l in labels} for label in labels}
        for (label1, label2), count in neighbor_stats.items():
            matrix[label1][label2] = count
        heatmap_data = [[matrix[label1][label2] for label2 in labels] for label1 in labels]

        sns.heatmap(heatmap_data, xticklabels=labels, yticklabels=labels, cmap='Blues', annot=True, ax=ax)
        ax.set_title(f"Entity Pair Frequencies ({title})")
        ax.set_xlabel("Next Entity")
        ax.set_ylabel("Previous Entity")
    
    plot_heatmap_on_ax(neighbor_train_data, axes[0], "Train")
    plot_heatmap_on_ax(neighbor_test_data, axes[1], "Test")
    plot_heatmap_on_ax(neighbor_valid_data, axes[2], "Valid")

    plt.tight_layout()
    plt.show()

def generate_wordcloud_from_tokens_on_subplots(*datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6))
    
    def generate_wordcloud(dataset, ax, title):

        text = " ".join([token for sentence in dataset for token, label in sentence if label != 'O'])
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)

    for ax, dataset, title in zip(axes, datasets, ['Train dataset', 'Test dataset', 'Valid dataset']):
        generate_wordcloud(dataset, ax, title)

    plt.tight_layout()
    plt.show()

def remove_duplicates(token_sentences: List[list]) -> List[list]:
    seen_sentences = set()
    unique_sentences = []
    for sentence in token_sentences:
        sentence_tuple = tuple(sentence)
        if sentence_tuple not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence_tuple)

    cleaned_sentences = []
    for sentence in unique_sentences:
        seen_tokens = set()
        cleaned_sentence = []
        for token, label in sentence:
            if token not in seen_tokens:
                cleaned_sentence.append((token, label))
                seen_tokens.add(token)
        cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences

def remove_noise_from_data(token_sentences: List[list]) -> List[list]:
    invalid_token_pattern = r'[^a-zA-Zа-яА-ЯёЁ0-9]'
    cleaned_dataset = []
    
    for sentence in token_sentences:
        cleaned_sentence = []
        for token, label in sentence:
            if re.search(invalid_token_pattern, token):
                continue

            if label in ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'] and token.isdigit():
                continue
        
            cleaned_sentence.append((token, label))

        cleaned_dataset.append(cleaned_sentence)

    return cleaned_dataset


def remove_stop_words_from_dataset(token_sentences: List[list]) -> List[list]:
    """
    Removes stop words from tokens in the dataset for the NER task.
    
    :param dataset: List of sentences (each sentence is a list of tuples (token, label))
    :return: New dataset without stopwords
    """
    cleaned_dataset = []

    for sentence in token_sentences:
        cleaned_sentence = []

        for token, label in sentence:
            if token.lower() not in stop_words:
                cleaned_sentence.append((token, label))
    
        if cleaned_sentence:
            cleaned_dataset.append(cleaned_sentence)

    return cleaned_dataset

def convert_to_lowercase_for_dataset(token_sentences: List[list]) -> List[list]:
    """
    Casts all tokens in the dataset to lowercase, leaving entity labels unchanged.
    
    :param dataset: List of sentences (each sentence is a list of tuples (token, label))
    :return: New dataset with tokens in lower case
    """
    lowercase_dataset = []

    for sentence in token_sentences:
        lowercase_sentence = [(token.lower(), label) for token, label in sentence]
        lowercase_dataset.append(lowercase_sentence)

    return lowercase_dataset

def remove_special_characters_from_dataset(token_sentences: List[list]) -> List[list]:
    """
    Removes special characters from tokens in the dataset, leaving entity labels unchanged.
    
    :param dataset: List of sentences (each sentence is a list of tuples (token, label))
    :return: New dataset with tokens without special characters
    """
    cleaned_dataset = []

    for sentence in token_sentences:
        cleaned_sentence = [(re.sub(r'[^A-Za-z0-9\s]', '', token), label) for token, label in sentence]
        cleaned_dataset.append(cleaned_sentence)

    return cleaned_dataset

def evaluate_ner(dataset):
    """
    Calculates metrics for NER: accuracy, precision, recall, f1-score.

    :param sample (dict): Dictionary with text, predicted labels, and target labels.
    :return dict: Metrics (accuracy, precision, recall, f1-score).
    """
    accuracy, precision, recall, f1 = [], [], [], []
    for sample in dataset:
        predicted = [label for _, label in sample["predicted"]]
        target = [label for _, label in sample["target"]]

        if len(predicted) != len(target):
            raise ValueError("Длины predicted и target должны совпадать.")

        accuracy.append(accuracy_score(target, predicted))
        precision.append(precision_score(target, predicted, average="macro", zero_division=0))
        recall.append(recall_score(target, predicted, average="macro", zero_division=0))
        f1.append(f1_score(target, predicted, average="macro", zero_division=0))

    return {
        "accuracy": np.mean(accuracy),
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "f1-score": np.mean(f1)
    }