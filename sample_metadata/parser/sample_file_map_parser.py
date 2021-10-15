# pylint: disable=too-many-instance-attributes,too-many-locals,unused-argument,no-self-use,wrong-import-order,unused-argument
from typing import List
import logging

from sample_metadata.parser.generic_metadata_parser import GenericMetadataParser

SAMPLE_ID_COL_NAME = 'Individual ID'
READS_COL_NAME = 'Filenames'

__DOC = """
The SampleFileMapParser is used for parsing files with format:

- 'Individual ID'
- 'Filenames'

EG:
    Individual ID   Filenames
    <sample-id>     <sample-id>.filename-R1.fastq.gz,<sample-id>.filename-R2.fastq.gz
    # OR
    <sample-id2>    <sample-id2>.filename-R1.fastq.gz
    <sample-id2>    <sample-id2>.filename-R2.fastq.gz

This format is useful for ingesting filenames for the seqr loading pipeline
"""

logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class SampleFileMapParser(GenericMetadataParser):
    """Parser for SampleFileMap"""

    def __init__(
        self,
        search_locations: List[str],
        sample_metadata_project: str,
        default_sequence_type='wgs',
        default_sample_type='blood',
        confirm=False,
    ):
        super().__init__(
            search_locations=search_locations,
            sample_metadata_project=sample_metadata_project,
            sample_name_column=SAMPLE_ID_COL_NAME,
            reads_column=READS_COL_NAME,
            default_sequence_type=default_sequence_type,
            default_sample_type=default_sample_type,
            sample_meta_map={},
            sequence_meta_map={},
            qc_meta_map={},
            confirm=confirm,
        )

    @staticmethod
    def from_manifest_path(
        manifest: str,
        sample_metadata_project: str,
        default_sequence_type='wgs',
        default_sample_type='blood',
        search_paths=None,
        confirm=False,
    ):
        """From manifest path, same defaults in __init__"""
        super().from_manifest_path(
            manifest=manifest,
            search_paths=search_paths,
            sample_metadata_project=sample_metadata_project,
            sample_name_column=SAMPLE_ID_COL_NAME,
            reads_column=READS_COL_NAME,
            default_sequence_type=default_sequence_type,
            default_sample_type=default_sample_type,
            sample_meta_map={},
            sequence_meta_map={},
            qc_meta_map={},
        )
