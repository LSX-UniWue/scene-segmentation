# Code and Data for the Paper "Assessing the State of the Art in Scene Segmentation"

## Code

Code is provided for Sequential Sentence Classification (`ssc`), LLM prompting (`prompting`) and LLM
fine-tuning (`llama`).

The prompting code requires providing an API key for OpenAI and potentially OpenRouter (for additional models) as well
as a BaseURL for a running Ollama server in `prompting/classify.py`.

## Data

The data folder contains full annotated files for the public domain texts in our corpus and stand-off annotations for
the other texts.

### Standoff Annotations

The standoff annotations are simple json files containing the character indices of scene boundaries as well as detected
sentence boundaries.
Additionally, each file contains a hash of the text (`md5(doc.text.encode("utf-8")).hexdigest()`), which can be used to
ensure that the text you are using matches our
annotations.
The format is as follows:

```json
{
  "scenes": [
    {
      "start": 0,
      "end": 100,
      "reason_for_change": "Zeit, Handlung",
      "scene_type": "Szene"
    },
    ...
  ],
  "sentences": [
    {
      "start": 0,
      "end": 10
    },
    ...
  ],
  "md5": "hash"
}
```

### Full Annotated Files

The full annotated files are in UIMA XMI format and can be viewed most easily by pulling them into the editor window
of [WebATHEN](https://webathen-beta.informatik.uni-wuerzburg.de/?view=annotationbrowserView). For automatic processing,
the easiest way is the use of the [WueNLP](https://gitlab2.informatik.uni-wuerzburg.de/kallimachos/wuenlp/) python
library:

```python
from wuenlp.impl.UIMANLPStructs import UIMADocument

doc = UIMADocument.from_xmi("path/to/file.xmi")

for scene in doc.scenes:
    print(scene.text)
    ...
```