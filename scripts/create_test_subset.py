#!/usr/bin/env python3
# pylint: disable=too-many-instance-attributes,too-many-locals,unused-argument,no-self-use,wrong-import-order,unused-argument,too-many-arguments

import os
import subprocess
import sys
import traceback
from collections import Counter
from typing import Dict, List, Optional, Tuple
import random
import logging
import click
from google.cloud import storage

from sample_metadata.apis import (
    AnalysisApi,
    SequenceApi,
    SampleApi,
    FamilyApi,
    ParticipantApi,
)
from sample_metadata.models import (
    NewSequence,
    NewSample,
    AnalysisModel,
    SampleUpdateModel,
    AnalysisType,
    AnalysisStatus,
    SampleType,
    SequenceType,
    SequenceStatus,
)
from sample_metadata import exceptions
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
    '-s',
    'sample_ids',
    multiple=True,
    help='Specific sample IDs to pull',
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
    sample_ids: Optional[List[str]],
    samples_n: Optional[int],
    families_n: Optional[int],
):
    """
    Script creates a test subset for a given project.
    A new project with a prefix -test is created, and for any files in sample/meta,
    sequence/meta, or analysis/output a copy in the -test namespace is created.
    """
    main_samples = _select_samples(sample_ids, samples_n, families_n, project)
    main_cpgids = [s['id'] for s in main_samples]

    # Populating test project
    target_project = project + '-test'
    logger.info('Checking any existing test samples in the target test project')

    test_sample_by_external_id = _process_existing_test_samples(
        target_project, main_samples
    )

    try:
        main_seq_infos: List[Dict] = seqapi.get_sequences_by_sample_ids(main_cpgids)
    except exceptions.ApiException:
        main_seq_by_cpgid = {}
    else:
        main_seq_by_cpgid = {seq['sample_id']: seq for seq in main_seq_infos}

    main_analysis_by_cpgid_by_type = {'cram': {}, 'gvcf': {}}
    for a_type, main_analysis_by_sid in main_analysis_by_cpgid_by_type.items():
        try:
            main_analyses: List[Dict] = aapi.get_latest_analysis_for_samples_and_type(
                project=project,
                analysis_type=a_type,
                request_body=main_cpgids,
            )
        except exceptions.ApiException:
            traceback.print_exc()
        else:
            for a in main_analyses:
                if a_type == 'cram':
                    a[
                        'output'
                    ] = f'gs://cpg-{project}-main/cram/{a["sample_ids"][0]}.cram'
                if a_type == 'gvcf':
                    a[
                        'output'
                    ] = f'gs://cpg-{project}-main/gvcf/{a["sample_ids"][0]}.g.vcf.gz'
                main_analysis_by_sid[a['sample_ids'][0]] = a
        logger.info(f'Will copy {a_type} analysis entries: {main_analysis_by_sid}')

    for main_s in main_samples:
        logger.info(f'Processing sample {main_s["id"]}')

        if main_s['external_id'] in test_sample_by_external_id:
            test_cpgid = test_sample_by_external_id.get(main_s['external_id'])['id']
            logger.info(f'Sample already in test project, with ID {test_cpgid}')
        else:
            logger.info('Creating test sample entry')
            test_cpgid = sapi.create_new_sample(
                project=target_project,
                new_sample=NewSample(
                    external_id=main_s['external_id'],
                    type=SampleType(main_s['type']),
                ),
            )
            sapi.update_sample(
                test_cpgid,
                SampleUpdateModel(
                    meta=_copy_files_in_dict(
                        main_s['meta'],
                        project,
                        old_cpgid=main_s['id'],
                        new_cpgid=test_cpgid,
                    )
                ),
            )
            seq_info = main_seq_by_cpgid.get(main_s['id'])
            if seq_info:
                logger.info('Processing sequence entry')
                new_meta = _copy_files_in_dict(
                    seq_info.get('meta'),
                    project,
                    old_cpgid=main_s['id'],
                    new_cpgid=test_cpgid,
                )
                logger.info('Creating sequence entry in test')
                seqapi.create_new_sequence(
                    new_sequence=NewSequence(
                        sample_id=test_cpgid,
                        meta=new_meta,
                        type=SequenceType(seq_info['type']),
                        status=SequenceStatus(seq_info['status']),
                    )
                )

        for a_type in ['cram', 'gvcf']:
            analysis = main_analysis_by_cpgid_by_type[a_type].get(main_s['id'])
            if analysis:
                logger.info(f'Processing {a_type} analysis entry')
                am = AnalysisModel(
                    output=_copy_files_in_dict(
                        analysis['output'],
                        project,
                        old_cpgid=main_s['id'],
                        new_cpgid=test_cpgid,
                    ),
                    type=AnalysisType(a_type),
                    status=AnalysisStatus(analysis['status']),
                    sample_ids=[test_cpgid],
                    meta=analysis['meta'],
                )
                logger.info(f'Creating {a_type} analysis entry in test')
                aapi.create_new_analysis(project=target_project, analysis_model=am)
        logger.info(f'-')


def _validate_opts(
    sample_ids, samples_n, families_n
) -> Tuple[Optional[List[str]], Optional[int], Optional[int]]:
    if samples_n is not None and families_n is not None and sample_ids is not None:
        raise click.BadParameter(
            'Please specify only one of --samples, -s, or --families '
            + '(though -s can be specified multiple times)'
        )

    if samples_n is None and families_n is None and sample_ids is None:
        samples_n = DEFAULT_SAMPLES_N
        logger.info(
            f'Neither --samples, -s, nor --families specified, defaulting to selecting '
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

    return sample_ids, samples_n, families_n


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


def _copy_files_in_dict(
    d, dataset: str, old_cpgid: Optional[str], new_cpgid: Optional[str]
):
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
        if old_cpgid and new_cpgid:
            new_path = new_path.replace(old_cpgid, new_cpgid)
        if not file_exists(new_path):
            cmd = f'gsutil cp "{old_path}" "{new_path}"'
            logger.info(f'Copying file in metadata: {cmd}')
            subprocess.run(cmd, check=False, shell=True)
        extra_exts = ['.md5']
        if new_path.endswith('.vcf.gz'):
            extra_exts.append('.tbi')
        if new_path.endswith('.cram'):
            extra_exts.append('.crai')
        for ext in extra_exts:
            if file_exists(old_path + ext) and not file_exists(new_path + ext):
                cmd = f'gsutil cp "{old_path + ext}" "{new_path + ext}"'
                logger.info(f'Copying extra file in metadata: {cmd}')
                subprocess.run(cmd, check=False, shell=True)
        return new_path
    if isinstance(d, list):
        return [_copy_files_in_dict(x, dataset, old_cpgid, new_cpgid) for x in d]
    if isinstance(d, dict):
        return {
            k: _copy_files_in_dict(v, dataset, old_cpgid, new_cpgid)
            for k, v in d.items()
        }
    return d


def _pretty_format_samples(samples: List[Dict]) -> str:
    return ', '.join(f"{s['id']}/{s['external_id']}" for s in samples)


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


def _select_samples(sample_ids, samples_n, families_n, project):
    sample_ids, samples_n, families_n = _validate_opts(
        sample_ids, samples_n, families_n
    )
    all_main_samples = sapi.get_samples(
        body_get_samples_by_criteria_api_v1_sample_post={
            'project_ids': [project],
            'active': True,
        }
    )
    logger.info(f'Found {len(all_main_samples)} samples')

    if sample_ids:
        # Selecting specific samples
        main_samples = [
            s
            for s in all_main_samples
            if (s['id'] in sample_ids or s['external_id'] in sample_ids)
        ]
    else:
        if samples_n and samples_n >= len(all_main_samples):
            resp = str(
                input(
                    f'Requesting {samples_n} samples which is >= '
                    f'than the number of available samples ({len(all_main_samples)}). '
                    f'The test project will be a full copy of the production project. '
                    f'Please confirm (y): '
                )
            )
            if resp.lower() != 'y':
                raise SystemExit()

        random.seed(42)  # for reproducibility

        pid_sid = papi.get_external_participant_id_to_internal_sample_id(project)
        main_cpgid_by_participant_id = dict(pid_sid)

        if families_n is not None:
            main_samples = _select_families(
                project, families_n, main_cpgid_by_participant_id, all_main_samples
            )

        else:
            main_samples = random.sample(all_main_samples, samples_n)
    logger.info(
        f'Subset to {len(main_samples)} samples (internal ID / external ID): '
        f'{_pretty_format_samples(main_samples)}'
    )
    return main_samples


def _select_families(
    project, families_n, main_cpgid_by_participant_id, all_main_samples
):
    ped_lines = export_ped_file(project, replace_with_participant_external_ids=True)
    ped = Ped(ped_lines)
    families = list(ped.families.values())
    logger.info(f'Found {len(families)} families, by size:')
    _print_fam_stats(families)

    if families_n > len(families):
        logger.critical(
            f'Requested more families than found ({families_n} > {len(families)})'
        )
        sys.exit(1)

    families = random.sample(families, families_n)
    logger.info(f'After subsetting to {len(families)} families:')
    _print_fam_stats(families)

    main_cpgids = []
    for fam in families:
        for main_s in fam.samples:
            main_cpgids.append(main_cpgid_by_participant_id[main_s.sample_id])
    main_samples = [s for s in all_main_samples if s['id'] in main_cpgids]
    return main_samples


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
