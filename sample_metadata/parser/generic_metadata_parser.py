# pylint: disable=R0904,too-many-instance-attributes,too-many-locals,unused-argument,wrong-import-order,unused-argument,too-many-arguments,unused-import
import asyncio
import logging
import re
import shlex
from functools import reduce
from io import StringIO
from itertools import groupby
from typing import Dict, List, Optional, Any, Tuple, Union

import click

from sample_metadata.model.sample_type import SampleType
from sample_metadata.model.sequence_status import SequenceStatus
from sample_metadata.model.sequence_type import SequenceType
from sample_metadata.parser.generic_parser import (
    GenericParser,
    GroupedRow,
    ParticipantMetaGroup,
    SampleMetaGroup,
    SequenceMetaGroup,
    SingleRow,
    run_as_sync,
)  # noqa

__DOC = """
Parse CSV / TSV manifest of arbitrary format.
This script allows you to specify HOW you want the manifest
to be mapped onto individual data.

This script loads the WHOLE file into memory

It groups rows by the sample ID, and collapses metadata from rows.

EG:
    Sample ID       sample-collection-date  depth  qc_quality  Fastqs

    <sample-id>     2021-09-16              30x    0.997       <sample-id>.filename-R1.fastq.gz,<sample-id>.filename-R2.fastq.gz

    # OR

    <sample-id2>    2021-09-16              30x    0.997       <sample-id2>.filename-R1.fastq.gz

    <sample-id2>    2021-09-16              30x    0.997       <sample-id2>.filename-R2.fastq.gz

Given the files are in a bucket called 'gs://cpg-upload-bucket/collaborator',
and we want to achieve the following:

- Import this manifest into the "$dataset" project of SM
- Map the following to `sample.meta`:
    - "sample-collection-date" -> "collection_date"
- Map the following to `sequence.meta`:
    - "depth" -> "depth"
    - "qc_quality" -> "qc.quality" (ie: {"qc": {"quality": 0.997}})
- Add a qc analysis object with the following mapped `analysis.meta`:
    - "qc_quality" -> "quality"

python parse_generic_metadata.py \
    --project $dataset \
    --sample-name-column "Sample ID" \
    --reads-column "Fastqs" \
    --sample-meta-field-map "sample-collection-date" "collection_date" \
    --sequence-meta-field "depth" \
    --sequence-meta-field-map "qc_quality" "qc.quality" \
    --qc-meta-field-map "qc_quality" "quality" \
    --search-path "gs://cpg-upload-bucket/collaborator" \
    <manifest-path>
"""


logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

RE_FILENAME_SPLITTER = re.compile('[,;]')


class GenericMetadataParser(GenericParser):
    """Parser for GenericMetadataParser"""

    def __init__(
        self,
        search_locations: List[str],
        participant_meta_map: Dict[str, str],
        sample_meta_map: Dict[str, str],
        sequence_meta_map: Dict[str, str],
        qc_meta_map: Dict[str, str],
        project: str,
        sample_name_column: str,
        participant_column: Optional[str] = None,
        reported_sex_column: Optional[str] = None,
        reported_gender_column: Optional[str] = None,
        karyotype_column: Optional[str] = None,
        reads_column: Optional[str] = None,
        checksum_column: Optional[str] = None,
        seq_type_column: Optional[str] = None,
        gvcf_column: Optional[str] = None,
        meta_column: Optional[str] = None,
        seq_meta_column: Optional[str] = None,
        batch_number: Optional[str] = None,
        reference_assembly_location_column: Optional[str] = None,
        default_reference_assembly_location: Optional[str] = None,
        default_sequence_type='genome',
        default_sequence_status='uploaded',
        default_sample_type='blood',
        allow_extra_files_in_search_path=False,
        **kwargs,
    ):
        super().__init__(
            path_prefix=None,
            search_paths=search_locations,
            project=project,
            default_sequence_type=default_sequence_type,
            default_sequence_status=default_sequence_status,
            default_sample_type=default_sample_type,
            **kwargs,
        )

        if not sample_name_column:
            raise ValueError('A sample name column MUST be provided')

        self.cpg_id_column = 'Internal CPG Sample ID'

        self.sample_name_column = sample_name_column
        self.participant_column = participant_column
        self.reported_sex_column = reported_sex_column
        self.reported_gender_column = reported_gender_column
        self.karyotype_column = karyotype_column
        self.seq_type_column = seq_type_column
        self.reference_assembly_location_column = reference_assembly_location_column
        self.default_reference_assembly_location = default_reference_assembly_location

        self.participant_meta_map = participant_meta_map or {}
        self.sample_meta_map = sample_meta_map or {}
        self.sequence_meta_map = sequence_meta_map or {}
        self.qc_meta_map = qc_meta_map or {}
        self.reads_column = reads_column
        self.checksum_column = checksum_column
        self.gvcf_column = gvcf_column
        self.meta_column = meta_column
        self.seq_meta_column = seq_meta_column
        self.allow_extra_files_in_search_path = allow_extra_files_in_search_path
        self.batch_number = batch_number

    def get_sample_id(self, row: SingleRow) -> Optional[str]:
        """Get external sample ID from row"""
        return row.get(self.sample_name_column, None)

    async def get_cpg_sample_id_from_row(self, row: SingleRow) -> Optional[str]:
        """Get internal cpg id from a row using get_sample_id and an api call"""
        return row.get(self.cpg_id_column, None)

    def get_sample_type(self, row: GroupedRow) -> SampleType:
        """Get sample type from row"""
        return SampleType(self.default_sample_type)

    def get_sequence_types(self, row: GroupedRow) -> List[SequenceType]:
        """
        Get sequence types from grouped row
        if SingleRow: return sequence type
        if GroupedRow: return sequence types for all rows
        """
        if isinstance(row, dict):
            return [self.get_sequence_type(row)]
        return [
            SequenceType(r.get(self.seq_type_column, self.default_sequence_type))
            for r in row
        ]

    def get_sequence_type(self, row: SingleRow) -> SequenceType:
        """Get sequence type from row"""
        value = row.get(self.seq_type_column, None) or self.default_sequence_type
        value = value.lower()

        if value == 'wgs':
            value = 'genome'
        elif value == 'wes':
            value = 'exome'
        elif 'mt' in value:
            value = 'mtseq'

        return SequenceType(value)

    def get_sequence_status(self, row: GroupedRow) -> SequenceStatus:
        """Get sequence status from row"""
        return SequenceStatus(self.default_sequence_status)

    def get_participant_id(self, row: SingleRow) -> Optional[str]:
        """Get external participant ID from row"""
        if not self.participant_column or self.participant_column not in row:
            raise ValueError('Participant column does not exist')
        return row[self.participant_column]

    def get_reported_sex(self, row: GroupedRow) -> Optional[int]:
        """Get reported sex from grouped row"""

        if not self.reported_sex_column:
            return None

        reported_sex = row[0].get(self.reported_sex_column, None)

        if reported_sex is None:
            return None
        if reported_sex == '':
            return None
        if reported_sex.lower() == 'female':
            return 2
        if reported_sex.lower() == 'male':
            return 1

        raise ValueError(
            f'{reported_sex} could not be identified as an input for reported_sex'
        )

    def get_reported_gender(self, row: GroupedRow) -> Optional[str]:
        """Get reported gender from grouped row"""
        return row[0].get(self.reported_gender_column, None)

    def get_karyotype(self, row: GroupedRow) -> Optional[str]:
        """Get karyotype from grouped row"""
        return row[0].get(self.karyotype_column, None)

    def has_participants(self, file_pointer, delimiter: str) -> bool:
        """Returns True if the file has a Participants column"""
        reader = self._get_dict_reader(file_pointer, delimiter=delimiter)
        first_line = next(reader)
        has_participants = self.participant_column in first_line
        file_pointer.seek(0)
        return has_participants

    async def validate_participant_map(
        self, participant_map: Dict[Any, Dict[str, List[Dict[str, Any]]]]
    ):
        await super().validate_participant_map(participant_map)
        if not self.reads_column:
            return

        ungrouped_rows: List[Dict[str, Any]] = []
        for sample_map in participant_map.values():
            for row in sample_map.values():
                if isinstance(row, list):
                    ungrouped_rows.extend(row)
                elif isinstance(row, dict):
                    ungrouped_rows.append(row)
                else:
                    raise ValueError(f'Unexpected type {type(row)} {row}')

        errors = []
        errors.extend(await self.check_files_covered_by_rows(ungrouped_rows))
        if errors:
            raise ValueError(', '.join(errors))

    async def validate_sample_map(self, sample_map: Dict[str, List[Dict[str, Any]]]):
        await super().validate_sample_map(sample_map)

        if not self.reads_column:
            return

        ungrouped_rows: List[Dict[str, Any]] = []
        for row in sample_map.values():
            if isinstance(row, list):
                ungrouped_rows.extend(row)
            elif isinstance(row, dict):
                ungrouped_rows.append(row)
            else:
                raise ValueError(f'Unexpected type {type(row)} {row}')

        errors = []
        errors.extend(await self.check_files_covered_by_rows(ungrouped_rows))
        if errors:
            raise ValueError(', '.join(errors))

    @staticmethod
    def flatten_irregular_list(irregular_list):
        """
        Flatten an irregular list: [1, [2, 3], 4]

        >>> GenericMetadataParser.flatten_irregular_list([1, [2, 3], [4,5]])
        [1, 2, 3, 4, 5]
        """
        return (
            [
                element
                for item in irregular_list
                for element in GenericMetadataParser.flatten_irregular_list(item)
            ]
            if isinstance(irregular_list, list)
            else [irregular_list]
        )

    async def get_all_files_from_row(self, sample_id: str, row):
        """Get all files from row, to allow subparsers to include other files"""
        fns = await self.get_read_filenames(sample_id, row)

        return self.flatten_irregular_list(fns)

    async def check_files_covered_by_rows(
        self, rows: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Check that the files in the search_paths are completely covered by the sample_map
        """
        filename_promises = []
        for grp in rows:
            for r in grp if isinstance(grp, list) else [grp]:
                filename_promises.append(
                    self.get_all_files_from_row(self.get_sample_id(r), r)
                )

        files_from_rows: List[str] = sum(await asyncio.gather(*filename_promises), [])
        filenames_from_rows = set(f.strip() for f in files_from_rows if f and f.strip())
        relevant_extensions = ('.cram', '.fastq.gz', '.bam')

        def filename_filter(f):
            return any(f.endswith(ext) for ext in relevant_extensions)

        file_from_search_paths = set(filter(filename_filter, self.filename_map.keys()))

        files_in_search_path_not_in_map = file_from_search_paths - filenames_from_rows
        missing_files = filenames_from_rows - file_from_search_paths

        errors = []

        if missing_files:
            errors.append(
                'There are files specified in the map, but not found in '
                f'the search paths: {", ".join(missing_files)}'
            )
        if files_in_search_path_not_in_map:
            m = (
                'There are files in the search path that are NOT covered by the file map: '
                f'{", ".join(files_in_search_path_not_in_map)}'
            )
            if self.allow_extra_files_in_search_path:
                logger.warning(m)
            else:
                errors.append(m)

        return errors

    @staticmethod
    def merge_dicts(a: Dict, b: Dict):
        """
        Recursively merge two dictionaries:
        - collapse equal values
        - put differing values into a list (not guaranteeing order)
        """
        if b is None:
            return a
        if a is None:
            return b

        res = {}
        for key in set(a.keys()).union(b.keys()):
            a_val = a.get(key)
            b_val = b.get(key)
            if a_val is not None and b_val is not None:
                # combine values
                a_is_dict = isinstance(a_val, dict)
                b_is_dict = isinstance(b_val, dict)

                if a_is_dict and b_is_dict:
                    # merge dict
                    res[key] = GenericMetadataParser.merge_dicts(a_val, b_val)
                elif a_val == b_val:
                    res[key] = a_val
                else:
                    res[key] = [a_val, b_val]
            else:
                res[key] = a_val or b_val

        return res

    @staticmethod
    def collapse_arbitrary_meta(key_map: Dict[str, str], row: GroupedRow):
        """
        This is a little bit tricky

        >>> GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new_key'}, {'key1': True})
        {'new_key': True}

        >>> GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new_key'}, [{'key1': True}, {'key1': True}])
        {'new_key': True}

        >>> GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new_key'}, [{'key1': True}, {'key1': None}])
        {'new_key': True}

        >>> GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new_key'}, [{'key1': True}])
        {'new_key': True}

        >>> GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new.key'}, [{'key1': True}])
        {'new': {'key': True}}

        >>> GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new.key'}, [{'key1': 1}, {'key1': 2}, {'key1': 3}])
        {'new': {'key': [1, 2, 3]}}

        # multiple keys sometimes is ordered, so check the sorted(dict.items())
        >>> import json; json.dumps(GenericMetadataParser.collapse_arbitrary_meta({'key1': 'new.key', 'key2': 'new.another'}, [{'key1': 1}, {'key1': 2}, {'key2': False}]), sort_keys=True)
        '{"new": {"another": false, "key": [1, 2]}}'
        """
        if not key_map or not row:
            return {}

        def prepare_dict_from_keys(key_parts: List[str], val):
            """Recursive production of dictionary"""
            if len(key_parts) == 1:
                return {key_parts[0]: val}
            return {key_parts[0]: prepare_dict_from_keys(key_parts[1:], val)}

        dicts = []
        for row_key, dict_key in key_map.items():
            if isinstance(row, list):
                inner_values = [r[row_key] for r in row if r.get(row_key) is not None]
                if any(isinstance(inner, list) for inner in inner_values):
                    # lists are unhashable
                    value = inner_values
                else:
                    value = list(set(inner_values))
                    if len(value) == 0:
                        continue
                    if len(value) == 1:
                        value = value[0]
            else:
                if row_key not in row:
                    continue
                value = row[row_key]

            dicts.append(prepare_dict_from_keys(dict_key.split('.'), value))

        return reduce(GenericMetadataParser.merge_dicts, dicts)

    @staticmethod
    def process_filename_value(string: Union[str, List[str]]) -> List[str]:
        """
        Split on multiple delimiters, ;,

        >>> GenericMetadataParser.process_filename_value('Filename1-fastq.gz;Filename2.fastq.gz')
        ['Filename1-fastq.gz', 'Filename2.fastq.gz']

        >>> GenericMetadataParser.process_filename_value('Filename1-fastq.gz, Filename2.fastq.gz')
        ['Filename1-fastq.gz', 'Filename2.fastq.gz']

        >>> GenericMetadataParser.process_filename_value('Filename1 with spaces fastq.gz')
        ['Filename1 with spaces fastq.gz']

        >>> GenericMetadataParser.process_filename_value(['filename ;filename2, filename3', ' filename4'])
        ['filename', 'filename2', 'filename3', 'filename4']
        """
        if not string:
            return []
        if isinstance(string, list):
            return sorted(
                set(
                    r
                    for f in string
                    for r in GenericMetadataParser.process_filename_value(f)
                )
            )

        filenames = [f.strip() for f in RE_FILENAME_SPLITTER.split(string)]
        filenames = [f for f in filenames if f]

        whitespace_filenames = [f for f in filenames if ' ' in f]
        if whitespace_filenames:
            logger.warning(
                'Whitespace detected in filenames: '
                + ','.join(shlex.quote(str(s)) for s in whitespace_filenames)
            )

        return filenames

    async def get_read_filenames(
        self, sample_id: Optional[str], row: SingleRow
    ) -> List[str]:
        """Get paths to reads from a row"""
        if not self.reads_column or self.reads_column not in row:
            return []
        # more post processing
        return self.process_filename_value(row[self.reads_column])

    async def get_checksums_from_row(
        self, sample_id: Optional[str], row: SingleRow, read_filenames: List[str]
    ) -> Optional[List[Optional[str]]]:
        """
        Get checksums for some row, you must either return:
            - no elements, or
            - number of elements equal to read_filenames

        Each element should be a string or None.
        """
        if not self.checksum_column or self.checksum_column not in row:
            return []

        return self.process_filename_value(row[self.checksum_column])

    async def get_gvcf_filenames(self, sample_id: str, row: GroupedRow) -> List[str]:
        """Get paths to gvcfs from a row"""
        if not self.gvcf_column:
            return []

        gvcf_filenames: List[str] = []
        for r in row if isinstance(row, list) else [row]:
            if self.gvcf_column in r:
                gvcf_filenames.extend(self.process_filename_value(r[self.gvcf_column]))

        return gvcf_filenames

    async def get_grouped_sample_meta(self, rows: GroupedRow) -> List[SampleMetaGroup]:
        """Return list of grouped by sample metadata from the rows"""
        sample_metadata = []
        for sid, row_group in groupby(rows, self.get_sample_id):
            sample_group = SampleMetaGroup(sample_id=sid, rows=row_group, meta=None)
            sample_metadata.append(await self.get_sample_meta(sample_group))
        return sample_metadata

    async def get_sample_meta(self, sample_group: SampleMetaGroup) -> SampleMetaGroup:
        """Get sample-metadata from row"""
        rows = sample_group.rows
        meta = self.collapse_arbitrary_meta(self.sample_meta_map, rows)
        sample_group.meta = meta
        return sample_group

    async def get_participant_meta(
        self, participant_id: int, rows: GroupedRow
    ) -> ParticipantMetaGroup:
        """Get participant-metadata from rows then set it in the ParticipantMetaGroup"""
        meta = self.collapse_arbitrary_meta(self.participant_meta_map, rows)
        return ParticipantMetaGroup(participant_id=participant_id, rows=rows, meta=meta)

    async def get_grouped_sequence_meta(
        self,
        sample_id: str,
        rows: GroupedRow,
    ) -> List[SequenceMetaGroup]:
        """
        Takes a collection of SingleRows and groups them by sequence type
        For each sequence type, get_sequence_meta for that group and return the
        resulting list of metadata
        """
        sequence_meta = []
        for stype, row_group in groupby(rows, self.get_sequence_type):
            seq_group = SequenceMetaGroup(
                rows=list(row_group),
                sequence_type=stype,
            )
            sequence_meta.append(await self.get_sequence_meta(seq_group, sample_id))
        return sequence_meta

    async def get_sequence_meta(
        self,
        seq_group: SequenceMetaGroup,
        sample_id: Optional[str] = None,
    ) -> SequenceMetaGroup:
        """Get sequence-metadata from row"""
        rows = seq_group.rows

        collapsed_sequence_meta = self.collapse_arbitrary_meta(
            self.sequence_meta_map, rows
        )

        read_filenames: List[str] = []
        gvcf_filenames: List[str] = []
        read_checksums: List[str] = []
        reference_assemblies: set[str] = set()

        for r in rows:
            _rfilenames = await self.get_read_filenames(sample_id=sample_id, row=r)
            read_filenames.extend(_rfilenames)
            if self.checksum_column and self.checksum_column in r:
                checksums = await self.get_checksums_from_row(sample_id, r, _rfilenames)
                if not checksums:
                    checksums = [None] * len(_rfilenames)
                read_checksums.extend(checksums)
            if self.gvcf_column and self.gvcf_column in r:
                gvcf_filenames.extend(self.process_filename_value(r[self.gvcf_column]))

            if self.reference_assembly_location_column:
                ref = r.get(self.reference_assembly_location_column)
                if ref:
                    reference_assemblies.add(ref)

        # strip in case collaborator put "file1, file2"
        full_read_filenames: List[str] = []
        full_gvcf_filenames: List[str] = []
        if read_filenames:
            full_read_filenames.extend(
                self.file_path(f.strip()) for f in read_filenames if f.strip()
            )
        if gvcf_filenames:
            full_gvcf_filenames.extend(
                self.file_path(f.strip()) for f in gvcf_filenames if f.strip()
            )

        if not sample_id:
            sample_id = await self.get_cpg_sample_id_from_row(rows[0])

        read_file_types: Dict[str, Dict[str, List]] = await self.parse_files(
            sample_id, full_read_filenames, read_checksums
        )
        variant_file_types: Dict[str, Dict[str, List]] = await self.parse_files(
            sample_id, full_gvcf_filenames, None
        )
        reads: Dict[str, List] = read_file_types.get('reads')
        variants: Dict[str, List] = variant_file_types.get('variants')
        if reads:
            keys = list(reads.keys())
            if len(keys) > 1:
                # 2021-12-14 mfranklin: In future we should return multiple
                #       sequence meta, and handle that in the generic parser
                raise ValueError(
                    f'Multiple types of reads found ({", ".join(keys)}), currently not supported'
                )

            reads_type = keys[0]
            collapsed_sequence_meta['reads_type'] = reads_type
            collapsed_sequence_meta['reads'] = reads[reads_type]

            if reads_type == 'cram':
                if len(reference_assemblies) > 1:
                    # sorted for consistent testing
                    str_ref_assemblies = ', '.join(sorted(reference_assemblies))
                    raise ValueError(
                        f'Multiple reference assemblies were defined for {sample_id}: {str_ref_assemblies}'
                    )
                if len(reference_assemblies) == 1:
                    ref = next(iter(reference_assemblies))
                else:
                    ref = self.default_reference_assembly_location

                if not ref:
                    raise ValueError(
                        f'Reads type for "{sample_id}" is CRAM, but a reference is not defined, please set the default reference assembly path'
                    )

                ref_fp = self.file_path(ref)
                secondary_files = (
                    await self.create_secondary_file_objects_by_potential_pattern(
                        ref_fp, ['.fai']
                    )
                )
                cram_reference = await self.create_file_object(
                    ref_fp, secondary_files=secondary_files
                )
                collapsed_sequence_meta['reference_assembly'] = cram_reference

        if variants:
            if 'gvcf' in variants:
                collapsed_sequence_meta['gvcfs'] = variants.get('gvcf')
                collapsed_sequence_meta['gvcf_types'] = 'gvcf'

            if 'vcf' in variants:
                collapsed_sequence_meta['vcfs'] = variants['vcf']
                collapsed_sequence_meta['vcf_type'] = 'vcf'

        if self.batch_number is not None:
            collapsed_sequence_meta['batch'] = self.batch_number

        seq_group.meta = collapsed_sequence_meta
        return seq_group

    async def get_qc_meta(
        self, sample_id: str, row: GroupedRow
    ) -> Optional[Dict[str, Any]]:
        """Get collapsed qc meta"""
        if not self.qc_meta_map:
            return None

        return self.collapse_arbitrary_meta(self.qc_meta_map, row)

    async def from_manifest_path(
        self,
        manifest: str,
        confirm=False,
        delimiter=None,
        dry_run=False,
    ):
        """Parse manifest from path, and return result of parsing manifest"""
        file = self.file_path(manifest)

        _delimiter = delimiter or GenericMetadataParser.guess_delimiter_from_filename(
            file
        )

        file_contents = await self.file_contents(file)
        return await self.parse_manifest(
            StringIO(file_contents),
            delimiter=_delimiter,
            confirm=confirm,
            dry_run=dry_run,
        )


@click.command(help=__DOC)
@click.option(
    '--project',
    required=True,
    help='The sample-metadata project ($DATASET) to import manifest into',
)
@click.option('--sample-name-column', required=True)
@click.option(
    '--reads-column',
    help='Column where the reads information is held, comma-separated if multiple',
)
@click.option(
    '--gvcf-column',
    help='Column where the reads information is held, comma-separated if multiple',
)
@click.option(
    '--qc-meta-field-map',
    nargs=2,
    multiple=True,
    help='Two arguments per listing, eg: --qc-meta-field "name-in-manifest" "name-in-analysis.meta"',
)
@click.option(
    '--participant-meta-field',
    multiple=True,
    help='Single argument, key to pull out of row to put in participant.meta',
)
@click.option(
    '--participant-meta-field-map',
    nargs=2,
    multiple=True,
    help='Two arguments per listing, eg: --participant-meta-field-map "name-in-manifest" "name-in-participant.meta"',
)
@click.option(
    '--sample-meta-field',
    multiple=True,
    help='Single argument, key to pull out of row to put in sample.meta',
)
@click.option(
    '--sample-meta-field-map',
    nargs=2,
    multiple=True,
    help='Two arguments per listing, eg: --sample-meta-field-map "name-in-manifest" "name-in-sample.meta"',
)
@click.option(
    '--sequence-meta-field',
    multiple=True,
    help='Single argument, key to pull out of row to put in sample.meta',
)
@click.option(
    '--sequence-meta-field-map',
    nargs=2,
    multiple=True,
    help='Two arguments per listing, eg: --sequence-meta-field "name-in-manifest" "name-in-sequence.meta"',
)
@click.option('--default-sample-type', default='blood')
@click.option('--default-sequence-type', default='wgs')
@click.option(
    '--confirm', is_flag=True, help='Confirm with user input before updating server'
)
@click.option('--search-path', multiple=True, required=True)
@click.argument('manifests', nargs=-1)
@run_as_sync
async def main(
    manifests,
    search_path: List[str],
    project,
    sample_name_column: str,
    participant_meta_field: List[str],
    participant_meta_field_map: List[Tuple[str, str]],
    sample_meta_field: List[str],
    sample_meta_field_map: List[Tuple[str, str]],
    sequence_meta_field: List[str],
    sequence_meta_field_map: List[Tuple[str, str]],
    qc_meta_field_map: List[Tuple[str, str]] = None,
    reads_column: Optional[str] = None,
    gvcf_column: Optional[str] = None,
    default_sample_type='blood',
    default_sequence_type='wgs',
    confirm=False,
):
    """Run script from CLI arguments"""
    if not manifests:
        raise ValueError('Expected at least 1 manifest')

    extra_seach_paths = [m for m in manifests if m.startswith('gs://')]
    if extra_seach_paths:
        search_path = list(set(search_path).union(set(extra_seach_paths)))

    participant_meta_map: Dict[Any, Any] = {}
    sample_meta_map: Dict[Any, Any] = {}
    sequence_meta_map: Dict[Any, Any] = {}

    qc_meta_map = dict(qc_meta_field_map or {})
    if participant_meta_field_map:
        participant_meta_map.update(dict(participant_meta_map))
    if participant_meta_field:
        participant_meta_map.update({k: k for k in participant_meta_field})
    if sample_meta_field_map:
        sample_meta_map.update(dict(sample_meta_field_map))
    if sample_meta_field:
        sample_meta_map.update({k: k for k in sample_meta_field})
    if sequence_meta_field_map:
        sequence_meta_map.update(dict(sequence_meta_field_map))
    if sequence_meta_field:
        sequence_meta_map.update({k: k for k in sequence_meta_field})

    parser = GenericMetadataParser(
        project=project,
        sample_name_column=sample_name_column,
        participant_meta_map=participant_meta_map,
        sample_meta_map=sample_meta_map,
        sequence_meta_map=sequence_meta_map,
        qc_meta_map=qc_meta_map,
        reads_column=reads_column,
        gvcf_column=gvcf_column,
        default_sample_type=default_sample_type,
        default_sequence_type=default_sequence_type,
        search_locations=search_path,
    )
    for manifest in manifests:
        logger.info(f'Importing {manifest}')

        await parser.from_manifest_path(manifest=manifest, confirm=confirm)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
