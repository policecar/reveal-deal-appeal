"""Probe fix: un-ignore ORG so presidio emits ORGANIZATION."""

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine

ner_config = NerModelConfiguration()
ner_config.labels_to_ignore = ner_config.labels_to_ignore - {"ORG", "ORGANIZATION"}

nlp_engine = SpacyNlpEngine(
    models=[{"lang_code": "en", "model_name": "en_core_web_lg"}],
    ner_model_configuration=ner_config,
)
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

text = (
    "Hi, this is John Smith from Acme Corporation calling about the Humanitec "
    "platform. We met with Microsoft in Berlin last week."
)
results = analyzer.analyze(
    text=text, language="en", entities=["PERSON", "ORGANIZATION", "LOCATION"]
)
for r in results:
    print(f"{r.entity_type:15} {text[r.start:r.end]!r}  score={r.score:.2f}")
