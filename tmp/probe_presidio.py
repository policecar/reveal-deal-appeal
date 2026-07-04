"""Probe: does the installed presidio-analyzer emit ORGANIZATION entities?"""

from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
text = (
    "Hi, this is John Smith from Acme Corporation calling about the Humanitec "
    "platform. We met with Microsoft in Berlin last week."
)
results = analyzer.analyze(
    text=text, language="en", entities=["PERSON", "ORGANIZATION", "LOCATION"]
)
for r in results:
    print(f"{r.entity_type:15} {text[r.start:r.end]!r}  score={r.score:.2f}")
print(
    f"\ntotal: {len(results)} | ORG count: {sum(r.entity_type == 'ORGANIZATION' for r in results)}"
)
