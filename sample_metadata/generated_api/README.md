# Sample Metadata Python API

This Python API is automatically generated through the `openapi3` annotations in combination with `openapi-generator`.
The script `/regenerate_api.py` should start the API server, and generate the annotations for you.

The Python API and isn't committed to the repository because CI system on each release should generate the Python API, and upload it to the CPG's conda repository.

There are no docs at the moment, but as a rough guide, you could do the following:

```python
from sample_metadata.api.samples_api import SamplesApi
sapi = SamplesApi()
sample = sapi.get_sample_by_external_id('dev', 'CPG0001')
print(sample)
# {'active': True,
#  'external_id': 'CPG0001',
#  'id': 1,
#  'meta': {'Volume (ul)': '100'},
#  'participant_id': None,
#  'type': 'blood'}
```

_This folder is contained within the `.gitignore`._
