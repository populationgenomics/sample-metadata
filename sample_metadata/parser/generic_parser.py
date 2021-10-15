# pylint: disable=too-many-instance-attributes,too-many-locals,unused-argument,no-self-use,assignment-from-none
import csv
import logging
import os
import re
from abc import abstractmethod
from collections import defaultdict
from itertools import groupby
from typing import List, Dict, Union, Optional, Tuple, Match

from google.api_core.exceptions import Forbidden

from sample_metadata.api import SampleApi, SequenceApi, AnalysisApi
from sample_metadata.models.new_sample import NewSample
from sample_metadata.models.new_sequence import NewSequence
from sample_metadata.models.sequence_type import SequenceType
from sample_metadata.models.sequence_update_model import SequenceUpdateModel
from sample_metadata.models.sample_update_model import SampleUpdateModel
from sample_metadata.models.analysis_model import AnalysisModel

logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

FASTQ_EXTENSIONS = ('.fq', '.fastq', '.fq.gz', '.fastq.gz')
BAM_EXTENSIONS = ('.bam',)
CRAM_EXTENSIONS = ('.cram',)
VCFGZ_EXTENSIONS = ('.vcf.gz',)

rmatch = re.compile(r'[_\.-][Rr]\d')
GroupedRow = Union[List[Dict[str, any]], Dict[str, any]]


class GenericParser:
    """Parser for VCGS manifest"""

    def __init__(
        self,
        path_prefix: str,
        sample_metadata_project: str,
        default_sequence_type='wgs',
        default_sample_type='blood',
        confirm=False,
    ):

        self.path_prefix = path_prefix
        self.confirm = confirm

        if not sample_metadata_project:
            raise ValueError('sample-metadata project is required')

        self.sample_metadata_project = sample_metadata_project

        self.default_sequence_type: str = default_sequence_type
        self.default_sample_type: str = default_sample_type

        # gs specific
        self.default_bucket = None

        self.client = None
        self.bucket_clients = {}

        if path_prefix and path_prefix.startswith('gs://'):
            # pylint: disable=import-outside-toplevel
            from google.cloud import storage

            self.client = storage.Client()
            path_components = path_prefix[5:].split('/')
            self.default_bucket = path_components[0]
            self.path_prefix = '/'.join(path_components[1:])

    def get_bucket(self, bucket_name=None):
        """Get cached bucket client from optional bucket name"""
        if bucket_name is None:
            bucket_name = self.default_bucket
        if bucket_name not in self.bucket_clients:
            self.bucket_clients[bucket_name] = self.client.get_bucket(bucket_name)

        return self.bucket_clients[bucket_name]

    def file_path(self, filename: str) -> str:
        """
        Get complete filepath of filename:
        - Includes gs://{bucket} if relevant
        - Includes path_prefix decided early on
        """
        if filename.startswith('gs://'):
            return filename

        if self.client and not filename.startswith('/'):
            return os.path.join(
                'gs://', self.default_bucket, self.path_prefix or '', filename
            )

        return os.path.join(self.path_prefix or '', filename)

    def get_blob(self, filename):
        """Convenience function for getting blob from fully qualified GCS path"""
        if not filename.startswith('gs://'):
            raise ValueError('No blob available')

        bucket_name, *components = filename[5:].split('/')
        bucket = self.get_bucket(bucket_name)
        path = '/'.join(components)

        # the next few lines are equiv to `bucket.get_blob(path)`
        # but without requiring storage.objects.get permission
        blobs = list(self.client.list_blobs(bucket, prefix=path))
        # first where r.name == path (or None)
        return next((r for r in blobs if r.name == path), None)

    def list_directory(self, directory_name) -> List[str]:
        """List directory"""
        path = self.file_path(directory_name)
        if path.startswith('gs://'):
            bucket_name, *components = directory_name[5:].split('/')
            blobs = self.client.list_blobs(bucket_name, prefix='/'.join(components))
            return [f'gs://{bucket_name}/{blob.name}' for blob in blobs]

        return [os.path.join(path, f) for f in os.listdir(path)]

    def file_contents(self, filename) -> Optional[str]:
        """Get contents of file (decoded as utf8)"""
        path = self.file_path(filename)
        if path.startswith('gs://'):
            blob = self.get_blob(filename)
            try:
                retval = blob.download_as_string()
                if isinstance(retval, bytes):
                    retval = retval.decode()
                return retval
            except Forbidden:
                logger.warning(f"FORBIDDEN: Can't download {filename}")
                return None

        with open(filename, encoding='utf-8') as f:
            return f.read()

    def file_exists(self, filename: str) -> bool:
        """Determines whether a file exists"""
        path = self.file_path(filename)

        if path.startswith('gs://'):
            blob = self.get_blob(filename)
            return blob is not None

        return os.path.exists(path)

    def file_size(self, filename):
        """Get size of file in bytes"""
        path = self.file_path(filename)
        if path.startswith('gs://'):
            blob = self.get_blob(filename)
            return blob.size

        return os.path.getsize(path)

    @abstractmethod
    def get_sample_id(self, row: Dict[str, any]) -> str:
        """Get external sample ID from row"""

    @abstractmethod
    def get_sample_meta(self, sample_id: str, row: GroupedRow) -> Dict[str, any]:
        """Get sample-metadata from row"""

    @abstractmethod
    def get_sequence_meta(self, sample_id: str, row: GroupedRow) -> Dict[str, any]:
        """Get sequence-metadata from row"""

    def get_qc_meta(self, sample_id: str, row: GroupedRow) -> Optional[Dict[str, any]]:
        """Get qc-meta from row, creates a QC object"""
        return None

    def get_sample_type(self, sample_id: str, row: GroupedRow) -> str:
        """Get sample type from row"""
        return self.default_sample_type

    def get_sequence_type(self, sample_id: str, row: GroupedRow) -> str:
        """Get sequence type from row"""
        return self.default_sequence_type

    @abstractmethod
    def get_sequence_status(self, sample_id: str, row: GroupedRow) -> str:
        """Get sequence status from row"""

    def parse_manifest(self, file_pointer, delimiter=',', dry_run=False):
        """
        Parse manifest from iterable (file pointer / String.IO)
        """
        # a sample has many rows
        sample_map = defaultdict(list)

        reader = csv.DictReader(file_pointer, delimiter=delimiter)
        for row in reader:
            sample_id = self.get_sample_id(row)
            sample_map[sample_id].append(row)

        # now we can start adding!!
        sapi = SampleApi()
        seqapi = SequenceApi()
        analysisapi = AnalysisApi()

        # determine if any samples exist
        external_id_map = sapi.get_sample_id_map_by_external(
            self.sample_metadata_project, list(sample_map.keys()), allow_missing=True
        )
        internal_sample_id_to_seq_id = {}

        if len(external_id_map) > 0:
            internal_sample_id_to_seq_id = seqapi.get_sequence_ids_from_sample_ids(
                request_body=list(external_id_map.values()),
            )

        samples_to_add: List[NewSample] = []
        # by external_sample_id
        sequences_to_add: Dict[str, List[NewSequence]] = defaultdict(list)
        qc_to_add: Dict[str, List[AnalysisModel]] = defaultdict(list)

        samples_to_update: Dict[str, SampleUpdateModel] = {}
        sequences_to_update: Dict[int, SequenceUpdateModel] = {}

        for external_sample_id in sample_map:
            logger.info(f'Preparing {external_sample_id}')
            rows = sample_map[external_sample_id]

            if len(rows) == 1:
                rows = rows[0]
            # now we have sample / sequencing meta across 4 different rows, so collapse them
            collapsed_sequencing_meta = self.get_sequence_meta(external_sample_id, rows)
            collapsed_sample_meta = self.get_sample_meta(external_sample_id, rows)
            collapsed_qc = self.get_qc_meta(external_sample_id, rows)

            sample_type = str(self.get_sample_type(external_sample_id, rows))
            sequence_status = self.get_sequence_status(external_sample_id, rows)

            if external_sample_id not in external_id_map:

                samples_to_add.append(
                    NewSample(
                        external_id=external_sample_id,
                        type=sample_type,
                        meta=collapsed_sample_meta,
                    )
                )

                if collapsed_qc:
                    qc_to_add[external_sample_id].append(
                        AnalysisModel(
                            sample_ids=['<none>'],
                            type='qc',
                            status='completed',
                            meta=collapsed_qc,
                        )
                    )
            else:
                # it already exists
                cpgid = external_id_map[external_sample_id]
                samples_to_update[cpgid] = SampleUpdateModel(
                    meta=collapsed_sample_meta,
                )
                # ignore QC results if sample already exists

            # should we add or update sequencing
            if (
                external_sample_id in external_id_map
                and external_id_map[external_sample_id] in sequences_to_update
            ):
                # update, we have a cpg sample ID AND a sequencing ID
                cpgid = external_id_map[external_sample_id]

                seq_id = internal_sample_id_to_seq_id[cpgid]
                sequences_to_update[seq_id] = SequenceUpdateModel(
                    meta=collapsed_sequencing_meta, status=sequence_status
                )
            else:
                # the internal sample ID gets mapped back later
                sequences_to_add[external_sample_id].append(
                    NewSequence(
                        sample_id='<None>',  # keep the type initialisation happy
                        meta=collapsed_sequencing_meta,
                        type=SequenceType.WGS,
                        status=sequence_status,
                    )
                )

        message = f"""\
Processing samples: {', '.join(sample_map.keys())}

Adding {len(samples_to_add)} samples
Adding {len(sequences_to_add)} sequences
Adding {len(qc_to_add)} QC analysis results

Updating {len(samples_to_update)} sample
Updating {len(sequences_to_update)} sequences"""

        if dry_run:
            logger.info('Dry run, so returning without inserting / updating metadata')
            return (
                samples_to_add,
                sequences_to_add,
                samples_to_update,
                sequences_to_update,
                qc_to_add,
            )

        if self.confirm:
            resp = str(input(message + '\n\nConfirm (y): '))
            if resp.lower() != 'y':
                raise SystemExit()
        else:
            logger.info(message)

        logger.info(f'Adding {len(samples_to_add)} samples to SM-DB database')
        ext_sample_to_internal_id = {}
        for new_sample in samples_to_add:
            sample_id = sapi.create_new_sample(
                project=self.sample_metadata_project, new_sample=new_sample
            )
            ext_sample_to_internal_id[new_sample.external_id] = sample_id

        for sample_id, sequences_to_add in sequences_to_add.items():
            for seq in sequences_to_add:
                seq.sample_id = ext_sample_to_internal_id[sample_id]
                seqapi.create_new_sequence(new_sequence=seq)

        for sample_id, qc_to_add in qc_to_add.items():
            for analysis in qc_to_add:
                analysis.sample_ids = [ext_sample_to_internal_id[sample_id]]
                analysisapi.create_new_analysis(
                    project=self.sample_metadata_project, analysis_model=analysis
                )

        logger.info(f'Updating {len(samples_to_update)} samples')
        for internal_sample_id, sample_update in samples_to_update.items():
            sapi.update_sample(
                id_=internal_sample_id,
                sample_update_model=sample_update,
            )

        logger.info(f'Updating {len(sequences_to_update)} sequences')
        for seq_id, seq_update in sequences_to_update.items():
            seqapi.update_sequence(
                sequence_id=seq_id,
                sequence_update_model=seq_update,
            )

        return ext_sample_to_internal_id

    def parse_file(
        self, reads: List[str]
    ) -> Tuple[Union[List[List[Dict]], List[Dict]], str]:
        """
        Returns a tuple of:
        1. single / list-of CWL file object(s), based on the extensions of the reads
        2. parsed type (fastq, cram, bam)
        """
        if all(any(r.lower().endswith(ext) for ext in FASTQ_EXTENSIONS) for r in reads):
            structured_fastqs = self.parse_fastqs_structure(reads)
            files = []
            for fastq_group in structured_fastqs:
                files.append([self.create_file_object(f) for f in fastq_group])

            return files, 'fastq'

        if all(any(r.lower().endswith(ext) for ext in CRAM_EXTENSIONS) for r in reads):
            sec_format = ['.crai', '^.crai']
            files = []
            for r in reads:
                secondaries = self.create_secondary_file_objects_by_potential_pattern(
                    r, sec_format
                )
                files.append(self.create_file_object(r, secondary_files=secondaries))

            return files, 'cram'

        if all(any(r.lower().endswith(ext) for ext in BAM_EXTENSIONS) for r in reads):
            sec_format = ['.bai', '^.bai']
            files = []
            for r in reads:
                secondaries = self.create_secondary_file_objects_by_potential_pattern(
                    r, sec_format
                )
                files.append(self.create_file_object(r, secondary_files=secondaries))

            return files, 'bam'

        if all(any(r.lower().endswith(ext) for ext in VCFGZ_EXTENSIONS) for r in reads):
            sec_format = ['.tbi']
            files = []
            for r in reads:
                secondaries = self.create_secondary_file_objects_by_potential_pattern(
                    r, sec_format
                )
                files.append(self.create_file_object(r, secondary_files=secondaries))

            return files, 'gvcf'

        extensions = set(os.path.splitext(r)[-1] for r in reads)
        joined_reads = ''.join(f'\n\t{i}: {r}' for i, r in enumerate(reads))
        raise ValueError(
            f'Mixed, or unrecognised extensions ({", ".join(extensions)}) for reads: {joined_reads}'
        )

    @staticmethod
    def parse_fastqs_structure(fastqs) -> List[List[str]]:
        """
        Takes a list of fastqs, and a set of nested lists of each R1 + R2 read.

        >>> GenericParser.parse_fastqs_structure(['/seqr_transfers/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L004_R1.fastq.gz', '/seqr_transfers/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L004_R2.fastq.gz'])
        [['/seqr_transfers/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L004_R1.fastq.gz', '/seqr_transfers/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L004_R2.fastq.gz']]

        >>> GenericParser.parse_fastqs_structure(['20210727_PROJECT1_L002_R2.fastq.gz', '20210727_PROJECT1_L002_R1.fastq.gz', '20210727_PROJECT1_L001_R2.fastq.gz', '20210727_PROJECT1_L001_R1.fastq.gz'])
        [['20210727_PROJECT1_L001_R1.fastq.gz', '20210727_PROJECT1_L001_R2.fastq.gz'], ['20210727_PROJECT1_L002_R1.fastq.gz', '20210727_PROJECT1_L002_R2.fastq.gz']]

        >>> GenericParser.parse_fastqs_structure(['/directory1/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L001_R1.fastq.gz', '/directory2/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L001_R2.fastq.gz'])
        [['/directory1/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L001_R1.fastq.gz', '/directory2/Z01_1234_HNXXXXX_TCATCCTT-AGCGAGCT_L001_R2.fastq.gz']]

        """
        # find last instance of R\d, and then group by prefix on that
        sorted_fastqs = sorted(fastqs)

        r_matches: Dict[str, Tuple[str, Optional[Match[str]]]] = {
            r: (os.path.basename(r), rmatch.search(os.path.basename(r)))
            for r in sorted_fastqs
        }
        no_r_match = [r for r, (_, matched) in r_matches.items() if matched is None]
        if no_r_match:
            no_r_match_str = ', '.join(no_r_match)
            raise ValueError(
                f"Couldn't detect the format of FASTQs (expected match for regex '{rmatch.pattern}'): {no_r_match_str}"
            )

        values = []
        for _, grouped in groupby(
            sorted_fastqs, lambda r: r_matches[r][0][: r_matches[r][1].start()]
        ):
            values.append(sorted(grouped))

        return sorted(values, key=lambda el: el[0])

    def create_file_object(
        self,
        filename: str,
        secondary_files: List[Dict[str, any]] = None,
    ) -> Dict[str, any]:
        """Takes filename, returns formed CWL dictionary"""
        checksum = None
        md5_filename = self.file_path(filename + '.md5')
        if self.file_exists(md5_filename):
            contents = self.file_contents(md5_filename)
            if contents:
                checksum = f'md5:{contents.strip()}'

        d = {
            'location': self.file_path(filename),
            'basename': os.path.basename(filename),
            'class': 'File',
            'checksum': checksum,
            'size': self.file_size(filename),
        }

        if secondary_files:
            d['secondaryFiles'] = secondary_files

        return d

    def create_secondary_file_objects_by_potential_pattern(
        self, filename, potential_secondary_patterns: List[str]
    ) -> List[Dict[str, any]]:
        """
        Take a base filename and potential secondary patterns:
        - Try each secondary pattern, see if it works
        - If it works, create a CWL file object
        - return a list of those secondary file objects that exist
        """
        secondaries = []
        for sec in potential_secondary_patterns:
            sec_file = _apply_secondary_file_format_to_filename(filename, sec)
            if self.file_exists(sec_file):
                secondaries.append(self.create_file_object(sec_file))

        return secondaries

    @staticmethod
    def guess_delimiter_from_filename(filename: str):
        """
        Guess delimiter from filename
        """
        extension_to_delimiter = {'.csv': ',', '.tsv': '\t'}
        relevant_delimiter = next(
            (
                delimiter
                for ext, delimiter in extension_to_delimiter.items()
                if filename.endswith(ext)
            ),
            None,
        )
        if relevant_delimiter:
            return relevant_delimiter

        raise ValueError(f'Unrecognised extension on file: {filename}')


def _apply_secondary_file_format_to_filename(
    filepath: Optional[str], secondary_file: str
):
    """
    You can trust this function to do what you want
    :param filepath: Filename to base
    :param secondary_file: CWL secondary format (Remove 1 extension for each leading ^).
    """
    if not filepath:
        return None

    fixed_sec = secondary_file.lstrip('^')
    leading = len(secondary_file) - len(fixed_sec)
    if leading <= 0:
        return filepath + fixed_sec

    basepath = ''
    filename = filepath
    if '/' in filename:
        idx = len(filepath) - filepath[::-1].index('/')
        basepath = filepath[:idx]
        filename = filepath[idx:]

    split = filename.split('.')

    newfname = filename + fixed_sec
    if len(split) > 1:
        newfname = '.'.join(split[: -min(leading, len(split) - 1)]) + fixed_sec
    return basepath + newfname
