"""Why no ORGANIZATION? Check spaCy raw NER vs presidio's entity mapping config."""

import spacy
from presidio_analyzer.nlp_engine import NerModelConfiguration

text = (
    "Hi, this is John Smith from Acme Corporation calling about the Humanitec "
    "platform. We met with Microsoft in Berlin last week."
)

nlp = spacy.load("en_core_web_lg")
print("spaCy raw:", [(e.text, e.label_) for e in nlp(text).ents])

c = NerModelConfiguration()
print("\ndefault labels_to_ignore:", c.labels_to_ignore)
print(
    "\nORG in mapping:",
    {k: v for k, v in c.model_to_presidio_entity_mapping.items() if "ORG" in k.upper()},
)
