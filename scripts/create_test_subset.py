#!/usr/bin/env python3
# pylint: disable=too-many-instance-attributes,too-many-locals,unused-argument,no-self-use,wrong-import-order,unused-argument,too-many-arguments

import os
import subprocess
from typing import Dict, List
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
    exceptions,
)

logger = logging.getLogger(__file__)
logging.basicConfig(format='%(levelname)s (%(name)s %(lineno)s): %(message)s')
logger.setLevel(logging.INFO)


sapi = SampleApi()
aapi = AnalysisApi()
seqapi = SequenceApi()


@click.command()
@click.option(
    '--project',
    required=True,
    help='The sample-metadata project ($DATASET)',
)
@click.option(
    '-n',
    required=True,
    type=int,
    default=10,
    help='Number of samples to subset',
)
def main(
    project: str,
    n: int,
):
    """
    Script creates a test subset for a given project.
    A new project with a prefix -test is created, and for any files in sample/meta,
    sequence/meta, or analysis/output a copy in the -test namespace is created.
    """
    if n < 1:
        raise click.BadParameter('Please specify n higher than 0')

    if n >= 100:
        resp = str(
            input(f'You requested a subset of {n} samples. Please confirm (y): ')
        )
        if resp.lower() != 'y':
            raise SystemExit()

    samples = sapi.get_samples(
        body_get_samples_by_criteria_api_v1_sample_post={
            'project_ids': [project],
            'active': True,
        }
    )
    logger.info(f'Found {len(samples)} samples')
    random.seed(42)  # for reproducibility
    samples = random.sample(samples, n)
    sample_ids = [s['id'] for s in samples]
    logger.info(
        f'Subset to {len(samples)} samples (internal ID / external ID): '
        f'{_pretty(samples)}'
    )

    # Checking existing test samples in the target test project
    target_project = project + '-test'
    test_sample_by_external_id = _process_existing_test_samples(target_project, samples)

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


def _pretty(samples: List[Dict]) -> str:
    return ', '.join('/'.join([s['id'], s['external_id']]) for s in samples)


def _process_existing_test_samples(test_project: str, samples: List) -> Dict:
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
        logger.info(f'Removing test samples: {_pretty(test_samples_to_remove)}')
        for s in test_samples_to_remove:
            sapi.update_sample(s['id'], SampleUpdateModel(active=False))

    if test_samples_to_keep:
        logger.info(f'Test samples already exist: {_pretty(test_samples_to_keep)}')

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


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
