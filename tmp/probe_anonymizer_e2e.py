"""End-to-end check of DatasetAnonymizer._clean_text after the ORG fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import DatasetAnonymizer

a = DatasetAnonymizer()
text = (
    "Hi, this is John Smith from Acme Corporation calling about the Humanitec "
    "platform. We met with Microsoft in Berlin last week."
)
print(a._clean_text(text))
