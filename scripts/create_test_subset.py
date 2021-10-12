#!/usr/bin/env python3
# pylint: disable=too-many-instance-attributes,too-many-locals,unused-argument,no-self-use,wrong-import-order,unused-argument,too-many-arguments

import os
import subprocess
import tempfile
from collections import Counter
from typing import Dict, List, Optional, Tuple
import random
import logging
import click
from google.cloud import storage

from sample_metadata import (
    AnalysisApi,
    SequenceApi,
    SampleApi,
    NewSequence,
    NewSample,
    AnalysisModel,
    SampleUpdateModel,
    FamilyApi,
    ParticipantApi,
    exceptions,
)
from sample_metadata.configuration import _get_google_auth_token
from peddy import Ped

logger = logging.getLogger(__file__)
logging.basicConfig(format='%(levelname)s (%(name)s %(lineno)s): %(message)s')
logger.setLevel(logging.INFO)

sapi = SampleApi()
aapi = AnalysisApi()
seqapi = SequenceApi()
fapi = FamilyApi()
papi = ParticipantApi()

DEFAULT_SAMPLES_N = 10


@click.command()
@click.option(
    '--project',
    required=True,
    help='The sample-metadata project ($DATASET)',
)
@click.option(
    '-n',
    '--samples',
    'samples_n',
    type=int,
    help='Number of samples to subset',
)
@click.option(
    '--families',
    'families_n',
    type=int,
    help='Minimal number of families to include',
)
def main(
    project: str,
    samples_n: Optional[int],
    families_n: Optional[int],
):
    """
    Script creates a test subset for a given project.
    A new project with a prefix -test is created, and for any files in sample/meta,
    sequence/meta, or analysis/output a copy in the -test namespace is created.
    """
    samples_n, families_n = _validate_opts(samples_n, families_n)

    samples = sapi.get_samples(
        body_get_samples_by_criteria_api_v1_sample_post={
            'project_ids': [project],
            'active': True,
        }
    )
    logger.info(f'Found {len(samples)} samples')
    if samples_n and samples_n >= len(samples):
        resp = str(
            input(
                f'Requesting {samples_n} samples which is >= '
                f'than the number of available samples ({len(samples)}). '
                f'The test project will be a copy of the production project. '
                f'Please confirm (y): '
            )
        )
        if resp.lower() != 'y':
            raise SystemExit()

    random.seed(42)  # for reproducibility

    ped_lines = export_ped_file(project, replace_with_participant_external_ids=True)
    if families_n is not None:
        ped = Ped(ped_lines)
        families = list(ped.families.values())
        logger.info(f'Found {len(families)} families, by size:')
        _print_fam_stats(families)
        families = random.sample(families, families_n)
        logger.info(f'After subsetting to {len(families)} families:')
        _print_fam_stats(families)
        p_ids = []
        for fam in families:
            for s in fam.samples:
                p_ids.append(s.sample_id)
        samples = [s for s in samples if s['external_id'] in p_ids]

    else:
        samples = random.sample(samples, samples_n)

    logger.info(
        f'Subset to {len(samples)} samples (internal ID / external ID): '
        f'{_pretty_format_samples(samples)}'
    )

    # Populating test project
    target_project = project + '-test'
    logger.info('Checking any existing test samples in the target test project')
    test_sample_by_external_id = _process_existing_test_samples(target_project, samples)

    sample_ids = [s['id'] for s in samples]
    try:
        seq_infos: List[Dict] = seqapi.get_sequences_by_sample_ids(sample_ids)
    except exceptions.ApiException:
        seq_info_by_s_id = {}
    else:
        seq_info_by_s_id = dict(zip(sample_ids, seq_infos))

    analysis_by_sid_by_type = {}
    for a_type in ['cram', 'gvcf']:
        try:
            analyses: List[Dict] = aapi.get_latest_analysis_for_samples_and_type(
                project=project,
                analysis_type=a_type,
                request_body=sample_ids,
            )
        except exceptions.ApiException:
            analysis_by_sid_by_type[a_type] = {}
        else:
            analysis_by_sid_by_type[a_type] = dict(zip(sample_ids, analyses))
            logger.info(
                f'Will copy {a_type} analysis entries: {analysis_by_sid_by_type[a_type]}'
            )

    for s in samples:
        logger.info(f'Processing sample {s["id"]}')

        if s['external_id'] in test_sample_by_external_id:
            new_s_id = test_sample_by_external_id.get(s['external_id'])['id']
        else:
            logger.info('Creating test sample entry')
            new_s_id = sapi.create_new_sample(
                project=target_project,
                new_sample=NewSample(
                    external_id=s['external_id'],
                    type=s['type'],
                    meta=_copy_files_in_dict(s['meta'], project),
                ),
            )

        seq_info = seq_info_by_s_id.get(s['id'])
        if seq_info:
            logger.info('Processing sequence entry')
            new_meta = _copy_files_in_dict(seq_info.get('meta'), project)
            logger.info('Creating sequence entry in test')
            seqapi.create_new_sequence(
                new_sequence=NewSequence(
                    sample_id=new_s_id,
                    meta=new_meta,
                    type=seq_info['type'],
                    status=seq_info['status'],
                )
            )

        for a_type in ['cram', 'gvcf']:
            analysis = analysis_by_sid_by_type[a_type].get(s['id'])
            if analysis:
                logger.info(f'Processing {a_type} analysis entry')
                am = AnalysisModel(
                    type=a_type,
                    output=_copy_files_in_dict(analysis['output'], project),
                    status=analysis['status'],
                    sample_ids=[s['id']],
                )
                logger.info(f'Creating {a_type} analysis entry in test')
                aapi.create_new_analysis(project=target_project, analysis_model=am)
        logger.info(f'-')

    logger.info('Exporting PED information')
    _import_ped_file_subset(target_project, ped_lines, samples)


def _validate_opts(samples_n, families_n) -> Tuple[Optional[int], Optional[int]]:
    if samples_n is not None and families_n is not None:
        raise click.BadParameter('Please specify only one of --samples or --families')

    if samples_n is None and families_n is None:
        samples_n = DEFAULT_SAMPLES_N
        logger.info(
            f'Neither --samples nor --families specified, defaulting to selecting '
            f'{samples_n} samples'
        )

    if samples_n is not None and samples_n < 1:
        raise click.BadParameter('Please specify --samples higher than 0')

    if families_n is not None and families_n < 1:
        raise click.BadParameter('Please specify --families higher than 0')

    if families_n is not None and families_n >= 30:
        resp = str(
            input(
                f'You requested a subset of {families_n} families. '
                f'Please confirm (y): '
            )
        )
        if resp.lower() != 'y':
            raise SystemExit()

    if samples_n is not None and samples_n >= 100:
        resp = str(
            input(
                f'You requested a subset of {samples_n} samples. '
                f'Please confirm (y): '
            )
        )
        if resp.lower() != 'y':
            raise SystemExit()
    return samples_n, families_n


def _print_fam_stats(families: List):
    fam_by_size = Counter()
    for fam in families:
        fam_by_size[len(fam.samples)] += 1
    for fam_size in sorted(fam_by_size):
        if fam_size == 1:
            label = 'singles'
        elif fam_size == 2:
            label = 'duos'
        elif fam_size == 3:
            label = 'trios'
        else:
            label = f'{fam_size} members'
        logger.info(f'  {label}: {fam_by_size[fam_size]}')


def _copy_files_in_dict(d, dataset: str):
    """
    Replaces all `gs://cpg-{project}-main*/` paths
    into `gs://cpg-{project}-test*/` and creates copies if needed
    If `d` is dict or list, recursively calls this function on every element
    If `d` is str, replaces the path
    """
    if not d:
        return d
    if isinstance(d, str) and d.startswith(f'gs://cpg-{dataset}-main'):
        old_path = d
        if not file_exists(old_path):
            logger.warning(f'File {old_path} does not exist')
            return d
        new_path = old_path.replace(
            f'gs://cpg-{dataset}-main', f'gs://cpg-{dataset}-test'
        )
        if not file_exists(new_path):
            cmd = f'gsutil cp "{old_path}" "{new_path}"'
            logger.info(f'Copying file in metadata: {cmd}')
            subprocess.run(cmd, check=False, shell=True)
        for suf in ['.tbi', '.md5']:
            if file_exists(old_path + suf) and not file_exists(new_path + suf):
                cmd = f'gsutil cp "{old_path + suf}" "{new_path + suf}"'
                logger.info(f'Copying extra file in metadata: {cmd}')
                subprocess.run(cmd, check=False, shell=True)
        return new_path
    if isinstance(d, list):
        return [_copy_files_in_dict(x, dataset) for x in d]
    if isinstance(d, dict):
        return {k: _copy_files_in_dict(v, dataset) for k, v in d.items()}
    return d


def _pretty_format_samples(samples: List[Dict]) -> str:
    return ', '.join('/'.join([s['id'], s['external_id']]) for s in samples)


def _process_existing_test_samples(test_project: str, samples: List) -> Dict:
    """
    Removes samples that need to be removed and returns those that need to be kept
    """
    test_samples = sapi.get_samples(
        body_get_samples_by_criteria_api_v1_sample_post={
            'project_ids': [test_project],
            'active': True,
        }
    )
    external_ids = [s['external_id'] for s in samples]
    test_samples_to_remove = [
        s for s in test_samples if s['external_id'] not in external_ids
    ]
    test_samples_to_keep = [s for s in test_samples if s['external_id'] in external_ids]
    if test_samples_to_remove:
        logger.info(
            f'Removing test samples: {_pretty_format_samples(test_samples_to_remove)}'
        )
        for s in test_samples_to_remove:
            sapi.update_sample(s['id'], SampleUpdateModel(active=False))

    if test_samples_to_keep:
        logger.info(
            f'Test samples already exist: {_pretty_format_samples(test_samples_to_keep)}'
        )

    return {s['external_id']: s for s in test_samples_to_keep}


def file_exists(path: str) -> bool:
    """
    Check if the object exists, where the object can be:
        * local file
        * local directory
        * Google Storage object
    :param path: path to the file/directory/object
    :return: True if the object exists
    """
    if path.startswith('gs://'):
        bucket = path.replace('gs://', '').split('/')[0]
        path = path.replace('gs://', '').split('/', maxsplit=1)[1]
        gs = storage.Client()
        return gs.get_bucket(bucket).get_blob(path)
    return os.path.exists(path)


def export_ped_file(  # pylint: disable=invalid-name
    project: str,
    replace_with_participant_external_ids: bool = False,
    replace_with_family_external_ids: bool = False,
) -> List[str]:
    """
    Generates a PED file for the project, returs PED file lines in a list
    """
    route = f'/api/v1/family/{project}/pedigree'
    opts = []
    if replace_with_participant_external_ids:
        opts.append('replace_with_participant_external_ids=true')
    if replace_with_family_external_ids:
        opts.append('replace_with_family_external_ids=true')
    if opts:
        route += '?' + '&'.join(opts)

    cmd = f"""\
        curl --location --request GET \
        'https://sample-metadata.populationgenomics.org.au{route}' \
        --header "Authorization: Bearer {_get_google_auth_token()}"
        """

    lines = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
    return lines


def _import_ped_file_subset(project: str, ped_lines: List, samples: List):
    """
    Imports the pedigree information into the `project`'s database
    for the sample subset (`samples`), using `ped_lines` that were read before.
    """
    id_by_external_id = dict()
    for s in samples:
        id_by_external_id[s['external_id']] = s['id']
    new_lines = []
    for line in ped_lines:
        items = line.split('\t')
        external_id = items[1]
        pat_id = items[2]
        mat_id = items[3]
        if external_id in id_by_external_id:
            items[1] = id_by_external_id[external_id]
            items[2] = id_by_external_id.get(pat_id, pat_id)
            items[3] = id_by_external_id.get(mat_id, mat_id)
            new_lines.append('\t'.join(items))
    fh = tempfile.NamedTemporaryFile(delete=False)  # pylint:disable=consider-using-with
    fh.writelines([(line + '\n').encode() for line in new_lines])
    fh.close()
    fapi.import_pedigree(project, fh.name)
    os.unlink(fh)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
