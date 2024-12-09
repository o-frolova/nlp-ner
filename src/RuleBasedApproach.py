import spacy
from spacy.matcher import Matcher


class RuleBasedNER:
    def __init__(self, language="en"):
        self.nlp = spacy.blank(language)
        self.matcher = Matcher(self.nlp.vocab)
        self._add_rules()

    def _add_rules(self):
        # LOC
        loc_pattern = [{"IS_TITLE": True}, {"IS_ALPHA": True, "OP": "?"}]
        self.matcher.add("LOC", [loc_pattern])

        # PER
        per_pattern = [{"IS_TITLE": True}, {"IS_ALPHA": True, "OP": "?"}]
        self.matcher.add("PER", [per_pattern])

        # ORG
        org_pattern = [
            {"IS_TITLE": True},
            {"IS_ALPHA": True},
            {"LOWER": {"in": ["inc", "ltd", "corp", "company"]}, "OP": "?"}
        ]
        self.matcher.add("ORG", [org_pattern])

        # MISC
        misc_pattern = [{"IS_TITLE": True}, {"IS_ALPHA": True, "OP": "?"}]
        self.matcher.add("MISC", [misc_pattern])

    def process_text(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)

        labels = ["O"] * len(doc)

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            for i in range(start, end):
                if i == start:
                    labels[i] = f"B-{label}"
                else:
                    labels[i] = f"I-{label}"
        return [(token.text, label) for token, label in zip(doc, labels)]

    def process_dataset(self, dataset):
        results = []
        for _, sample in enumerate(dataset):
            text = sample["text"]
            target = sample["target"]

            doc = self.nlp(text)
            if len(doc) != len(target):
                continue

            predicted = self.process_text(text)
            results.append({"text": text, "predicted": predicted, "target": target})
        return results

