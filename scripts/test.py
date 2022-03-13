""" Basic Test Script """
from sample_metadata.apis import (
    SampleApi,
)

project = 'acute-care'
sapi = SampleApi()
samples = sapi.get_all_sample_id_map_by_internal(project)
print(samples)
