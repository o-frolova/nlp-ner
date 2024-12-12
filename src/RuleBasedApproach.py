import spacy
from spacy.matcher import Matcher
from sklearn.metrics import precision_recall_fscore_support

class RuleBasedNER:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        self.matcher = Matcher(self.nlp.vocab)
        self.ban_words_person = [
            "masters", "day", "week", "month", "free", "trade", "agreement", "north", "northern", 
            "south", "southern", "west", "western", "east", "eastern", "union", "senate", "state", 
            "department", "court", "house", "lake", "river", "mountain", "road", "beach", "bank", 
            "united", "x", "new", "city", "fire", "world", "cup", "earth", "corporation", "corp", 
            "corp.", "inc", "geological", "survey"
        ]
        self._add_rules()

    def _add_person_patterns(self):
        patterns_person = []
        patterns_person.append([{"TEXT": {"NOT_IN": self.ban_words_person}, "IS_ALPHA": True, "POS": "PROPN"}])

        patterns_person.append([
            {"POS": "PROPN", "LOWER": {"NOT_IN": self.ban_words_person}}, 
            {"POS": "PROPN", "LOWER": {"NOT_IN": self.ban_words_person}}
        ])

        patterns_person.append([
            {"LOWER": {"IN": ["mr", "president", "dr"]}},  # Добавляем "dr" в паттерн
            {"TEXT": "."},
            {"POS": "PROPN"}
        ])

        self.matcher.add("PER", patterns_person, greedy="LONGEST")

    def _add_organization_patterns(self):
        patterns_organisation = []

        patterns_organisation.append([
            {"POS": "PROPN"}, {"TEXT": {"REGEX": "(Inc|Corp|LLC|Ltd)"}}
        ])


        patterns_organisation.append([
            {"TEXT": {"IN": ["United", "European", "Global"]}}, 
            {"TEXT": {"IN": ["Union", "Council", "Group", "Agency"]}}
        ])

        patterns_organisation.append([
            {"TEXT": {"IN": ["UN", "FIFA", "Google", "Amazon", "Microsoft", "NASA"]}}
        ])

        patterns_organisation.append([
            {"POS": "PROPN"}, 
            {"POS": "PROPN", "LOWER": {"IN": ["department", "committee", "organization"]}}
        ])

        self.matcher.add("ORG", patterns_organisation, greedy="LONGEST")

    def _add_organization_patterns(self):
        possible_words_org = [
            "United", "Fire", "White", "European", "Nations", "Department",
            "House", "Union", "State", "Commission", "Corruption", "Anti", "-"
        ]

        patterns_organisation = [
           
            [{"POS": "PROPN"}, {"LOWER": {"REGEX": "^.*(inc|corp).*$"}}],
            [{"LOWER": "al"}, {"LOWER": "-"}, {"LOWER": "qaida"}],
            [{"TEXT": {"IN": ["Taliban", "Taleban", "Crips", "IAEA", "KCNA", "Nias",
                              "Sumatra", "Yemen", "US", "EU", "FIFA", "Senate"]}}],
            [{"TEXT": {"IN": possible_words_org}}, {"TEXT": {"IN": possible_words_org}}],
        ]

        self.matcher.add("ORG", patterns_organisation, greedy="LONGEST")
    
    def _add_location_patterns(self):
        
        possible_words_loc = ["Mount", "Lake", "River", "Beach", "Valley", "Hill", "Forest", "Park", "Street", "Avenue", "Boulevard", "Road", "Square", "Plaza"]
        patterns_location = []
        patterns_location.append([{"POS": "PROPN"}, {"TEXT": {"IN": possible_words_loc}}])
        patterns_location.append([{"POS": "PROPN"}, {"POS": "PROPN"}, {"TEXT": {"IN": possible_words_loc}}])
        patterns_location.append([{"TEXT": {"IN": ["Berlin", "Paris", "London", "Tokyo", "California", "Amazon", "Himalayas"]}}])
        self.matcher.add("LOC", patterns_location, greedy="LONGEST")
    
    def add_misc_patterns(self):
        possible_words_misc = ["Olympics", "World", "Cup", "Treaty", "Summit", "Agreement", "Championship", "Festival"]

        patterns_misc = []
        patterns_misc.append([{"TEXT": {"IN": possible_words_misc}}, {"TEXT": {"IN": possible_words_misc}}])
        patterns_misc.append([{"TEXT": {"IN": ["Nobel", "Grammy", "Oscar", "Pulitzer", "Emmy"]}}])
        patterns_misc.append([{"IS_TITLE": True}, {"IS_TITLE": True}])

        self.matcher.add("MISC", patterns_misc, greedy="LONGEST")  # добавление правил MISC

    def _add_rules(self):
        self._add_location_patterns()
        self._add_organization_patterns()
        self._add_person_patterns()
        self.add_misc_patterns()

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

    def process_dataset(self, dataset, model_name="en_core_web_sm"):
        nlp_standard = spacy.load(model_name)

        standard_results = []
        rule_based_results = []

        for sample in dataset:
            text = sample["text"]
            target = sample["target"]

            doc_standard = nlp_standard(text)
            predicted_standard = [(token.text, token.ent_iob_ + "-" + token.ent_type_ if token.ent_type_ else "O") for token in doc_standard]
            if len(target) != len(predicted_standard):
                continue
            standard_results.append({"prediction": predicted_standard, "reference": target})

            # Модель с правилами
            predicted_rules = self.process_text(text)
            if len(target) != len(predicted_rules):
                continue
            rule_based_results.append({"prediction": predicted_rules, "reference": target})

        return standard_results, rule_based_results

    def evaluate(self, dataset, model_name="en_core_web_sm"):
        standard_results, rule_based_results = self.process_dataset(dataset, model_name)

        y_true = []
        y_pred_standard = []
        y_pred_rules = []

        for standard_result, rule_result in zip(standard_results, rule_based_results):
            reference = standard_result["reference"]
            prediction_standard = standard_result["prediction"]
            prediction_rules = rule_result["prediction"]

            y_true.extend([label for _, label in reference])
            y_pred_standard.extend([label for _, label in prediction_standard])
            y_pred_rules.extend([label for _, label in prediction_rules])

        # Метрики для стандартной модели
        precision_standard, recall_standard, f1_standard, _ = precision_recall_fscore_support(
            y_true, y_pred_standard, average="weighted", zero_division=0
        )

        # Метрики для модели с правилами
        precision_rules, recall_rules, f1_rules, _ = precision_recall_fscore_support(
            y_true, y_pred_rules, average="weighted", zero_division=0
        )

        metrics = {
            "original model": {
                "precision": precision_standard,
                "recall": recall_standard,
                "f1": f1_standard,
            },
            "modified model": {
                "precision": precision_rules,
                "recall": recall_rules,
                "f1": f1_rules,
            },
        }

        return metrics
