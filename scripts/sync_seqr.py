import csv
import io
import os
import re
import json
import datetime
import logging
from typing import Any

import requests
from cloudpathlib import AnyPath
from sample_metadata.api.analysis_api import AnalysisApi

from sample_metadata.model.export_type import ExportType

from sample_metadata.apis import SeqrApi, ProjectApi

logging.basicConfig(level=logging.DEBUG)

MAP_LOCATION = 'gs://cpg-seqr-main-analysis/automation/'

SEQR_AUDIENCE = (
    '1021400127367-40kj6v68nlps6unk6bgvh08r5o4djf6b.apps.googleusercontent.com'
)
# SEQR_AUDIENCE = (
#     '1021400127367-9uc4sikfsm0vqo38q1g6rclj91mm501r.apps.googleusercontent.com'
# )

BASE = os.getenv('SEQR_URL', 'https://seqr-staging.populationgenomics.org.au')
url_individuals_table_upload = '/api/project/{projectGuid}/upload_individuals_table/sa'
url_individuals_table_confirm = (
    '/api/project/{projectGuid}/save_individuals_table/{uploadedFileId}/sa'
)

url_individual_metadata_table_upload = (
    '/api/project/{projectGuid}/upload_individuals_metadata_table/sa'
)
url_individual_metadata_table_confirm = (
    '/api/project/{projectGuid}/save_individuals_metadata_table/{uploadedFileId}/sa'
)

url_family_table_upload = '/api/project/{projectGuid}/upload_families_table/sa'
url_family_table_confirm = '/api/project/{projectGuid}/edit_families/sa'

url_update_es_index = '/api/project/{projectGuid}/add_dataset/variants/sa'
url_update_saved_variants = '/api/project/{projectGuid}/update_saved_variant_json/sa'

seqapi = SeqrApi()
papi = ProjectApi()
aapi = AnalysisApi()

ES_INDICES = {
    'acute-care': ['acute-care-genome-2022_0620_1843_l4h8u', 'acute-care-exome-2022_0624_0015_zs7fb'],
    'ravenscroft-arch': 'ravenscroft-arch-genome-2022_0618_1137_4qfyn',
    'circa': 'circa-genome-2022_0618_1137_4qfyn',
    'ohmr3-mendelian': 'ohmr3-mendelian-genome-2022_0618_1137_4qfyn',
    'validation': 'validation-genome-2022_0618_1137_4qfyn',
    'mito-disease': 'mito-disease-genome-2022_0618_1137_4qfyn',
    'perth-neuro': 'perth-neuro-genome-2022_0618_1137_4qfyn',
    'ohmr4-epilepsy': 'ohmr4-epilepsy-genome-2022_0618_1137_4qfyn',
    'hereditary-neuro': [
        'hereditary-neuro-genome-2022_0618_1137_4qfyn',
'hereditary-neuro-exome-2022_0624_0015_zs7fb'],
    'ravenscroft-rdstudy': 'ravenscroft-rdstudy-genome-2022_0618_1137_4qfyn',
    'heartkids': 'heartkids-genome-2022_0618_1137_4qfyn',
}


def sync_dataset(dataset: str, seqr_guid: str):
    """
    Synchronisation driver for a single dataset
    """
    # sync people first
    token = get_token()
    headers: dict[str, str] = {'Authorization': f'Bearer {token}'}
    params: dict[str, Any] = dict(
        dataset=dataset, project_guid=seqr_guid, headers=headers
    )

    sync_pedigree(**params)
    sync_families(**params)
    sync_individual_metadata(**params)
    update_es_index(**params)

    get_cram_map(dataset)



def sync_pedigree(dataset, project_guid, headers):
    """
    Synchronise pedigree from SM -> seqr in 3 steps:

    1. Get pedigree from SM
    2. Upload pedigree to seqr
    3. Confirm the upload
    """

    # 1. Get pedigree from SM
    pedigree_data = _get_pedigree_csv_from_sm(dataset)
    if not pedigree_data:
        return print(f'{dataset} :: Not updating pedigree because not data was found')

    # 2. Upload pedigree to seqr

    # use a filename ending with .csv to signal to seqr it's comma-delimited
    files = {'datafile': ('file.csv', pedigree_data)}
    req1_url = BASE + url_individuals_table_upload.format(projectGuid=project_guid)
    resp_1 = requests.post(req1_url, files=files, headers=headers)
    logging.debug(
        f'{dataset} :: Uploaded pedigree data with status: {resp_1.status_code}'
    )
    if not resp_1.ok:
        logging.warning(f'{dataset} :: Uploading pedigree failed: {resp_1.text}')
    resp_1.raise_for_status()
    resp_1_json = resp_1.json()
    if not resp_1_json or 'uploadedFileId' not in resp_1_json:
        raise ValueError(
            f'{dataset} :: Could not get uploadedFileId from {resp_1_json}'
        )
    uploaded_file_id = resp_1_json['uploadedFileId']

    # 3. Confirm the upload
    req2_url = BASE + url_individuals_table_confirm.format(
        projectGuid=project_guid, uploadedFileId=uploaded_file_id
    )
    is_ok = False
    remaining_attempts = 5
    while not is_ok:
        resp_2 = requests.post(req2_url, json=resp_1_json, headers=headers)
        is_ok = resp_2.ok
        if not is_ok:
            logging.warning(
                f'{dataset} :: Confirming pedigree failed, retrying '
                f'({remaining_attempts} more times): {resp_2.text}'
            )
            remaining_attempts -= 1
        if remaining_attempts <= 0:
            resp_2.raise_for_status()

    print(f'{dataset} :: Uploaded pedigree')


def sync_families(dataset, project_guid: str, headers: dict[str, str]):
    """
    Synchronise families template from SM -> seqr in 3 steps:

        1. Get family from SM
        2. Upload pedigree to seqr
        3. Confirm the upload
    """
    logging.debug(f'{dataset} :: Uploading family template')

    def _get_families_csv_from_sm(dataset: str):

        fam_rows = seqapi.get_families(project=dataset)
        fam_row_headers = [
            'Family ID',
            'Display Name',
            'Description',
            'Coded Phenotype',
        ]
        formatted_fam_rows = [','.join(fam_row_headers)]
        keys = ['external_id', 'external_id', 'description', 'coded_phenotype']
        formatted_fam_rows.extend(
            [','.join(str(r[k] or '') for k in keys) for r in fam_rows]
        )

        return '\n'.join(formatted_fam_rows)

    # 1. Get family data from SM
    family_data = _get_families_csv_from_sm(dataset)

    # use a filename ending with .csv to signal to seqr it's comma-delimited
    files = {'datafile': ('file.csv', family_data)}
    req1_url = BASE + url_family_table_upload.format(projectGuid=project_guid)
    resp_1 = requests.post(req1_url, files=files, headers=headers)
    logging.debug(
        f'{dataset} :: Uploaded new family template with status: {resp_1.status_code}'
    )
    if not resp_1.ok:
        logging.warning(f'{dataset} :: Request failed with information: {resp_1.text}')
    resp_1.raise_for_status()
    resp_1_json = resp_1.json()

    # 3. Confirm uploaded file

    req2_url = BASE + url_family_table_confirm.format(projectGuid=project_guid)
    is_ok = False
    remaining_attempts = 5
    while not is_ok:
        resp_2 = requests.post(req2_url, json=resp_1_json, headers=headers)
        is_ok = resp_2.ok
        if not is_ok:
            logging.warning(
                f'{dataset} :: Confirming family import failed: {resp_2.text}'
            )
            remaining_attempts -= 1
        if remaining_attempts <= 0:
            resp_2.raise_for_status()

    print(f'{dataset} :: Uploaded family template')


def sync_individual_metadata(dataset, project_guid, headers):

    individual_metadata_resp = seqapi.get_individual_metadata_for_seqr(
        project=dataset, export_type=ExportType('json')
    )

    if individual_metadata_resp is None:
        print(
            f'{dataset} :: There is an issue with getting individual metadata from SM, please try again later'
        )
        return

    json_rows = individual_metadata_resp['rows']
    individual_meta_headers = individual_metadata_resp['headers']
    col_header_map = individual_metadata_resp['header_map']

    if len(json_rows) == 0:
        print(f'{dataset} :: No individual metadata to sync')
        return

    output = io.StringIO()
    writer = csv.writer(output, delimiter=',')
    rows = [
        [col_header_map[h] for h in individual_meta_headers],
        *[[row.get(kh, '') for kh in individual_meta_headers] for row in json_rows],
    ]
    writer.writerows(rows)

    req1_url = BASE + url_individual_metadata_table_upload.format(
        projectGuid=project_guid
    )
    resp_1 = requests.post(
        req1_url, files={'datafile': ('file.csv', output.getvalue())}, headers=headers
    )
    print(
        f'{dataset} :: Uploaded individual metadata with status: {resp_1.status_code}'
    )
    if not resp_1.ok:
        print(f'{dataset} :: Request failed with information: {resp_1.text}')
    resp_1.raise_for_status()

    resp_1_json = resp_1.json()
    if not resp_1_json or 'uploadedFileId' not in resp_1_json:
        raise ValueError(
            f'{dataset} :: Could not get uploadedFileId from {resp_1_json}'
        )

    uploaded_file_id = resp_1_json['uploadedFileId']

    req2_url = BASE + url_individual_metadata_table_confirm.format(
        projectGuid=project_guid, uploadedFileId=uploaded_file_id
    )
    is_ok = False
    remaining_attempts = 5
    while not is_ok:
        resp_2 = requests.post(req2_url, json=resp_1_json, headers=headers)
        is_ok = resp_2.ok
        if not is_ok:
            logging.warning(
                f'{dataset} :: Confirming individual metadata import failed: {resp_2.text}'
            )
            remaining_attempts -= 1
        if remaining_attempts <= 0:
            resp_2.raise_for_status()

    print(f'{dataset} :: Uploaded individual metadata')


def update_es_index(dataset, project_guid, headers):

    person_sample_map_rows = seqapi.get_external_participant_id_to_internal_sample_id(
        project=dataset
    )

    rows_to_write = ['\t'.join(s[::-1]) for s in person_sample_map_rows]

    filename = f'{dataset}_pid_sid_map_{datetime.datetime.now().isoformat()}.tsv'
    filename = re.sub(r'[/\\?%*:|\'<>\x7F\x00-\x1F]', '-', filename)

    fn_path = os.path.join(MAP_LOCATION, filename)
    # pylint: disable=no-member
    with AnyPath(fn_path).open('w+') as f:
        f.write('\n'.join(rows_to_write))

    dataset_es_indices = ES_INDICES[dataset]
    if not isinstance(dataset_es_indices, list):
        dataset_es_indices = [dataset_es_indices]

    for es_index in dataset_es_indices:

        data = {
            'elasticsearchIndex': es_index,
            'datasetType': 'VARIANTS',
            'mappingFilePath': fn_path,
            'ignoreExtraSamplesInCallset': False,
        }
        print(data)

        req1_url = BASE + url_update_es_index.format(projectGuid=project_guid)
        resp_1 = requests.post(req1_url, json=data, headers=headers)
        print(f'{dataset} :: Updated ES index with status: {resp_1.status_code}')
        if not resp_1.ok:
            print(f'{dataset} :: Request failed with information: {resp_1.text}')
        resp_1.raise_for_status()

        req2_url = BASE + url_update_saved_variants.format(projectGuid=project_guid)
        resp_2 = requests.post(req2_url, json={}, headers=headers)
        print(f'{dataset} :: Updated saved variants with status code: {resp_2.status_code}')
        if not resp_2.ok:
            print(f'{dataset} :: Request failed with information: {resp_2.text}')
        resp_2.raise_for_status()


def _get_pedigree_csv_from_sm(dataset: str) -> str | None:
    """Call get_pedigree and return formatted string with header"""

    ped_rows = seqapi.get_pedigree(project=dataset)
    if not ped_rows:
        return None

    formatted_ped_rows = [
        'Family ID,Individual ID,Paternal ID,Maternal ID,Sex,Affected Status,Notes'
    ]
    keys = [
        'family_id',
        'individual_id',
        'paternal_id',
        'maternal_id',
        'sex',
        'affected',
        'notes',
    ]
    formatted_ped_rows.extend(','.join(str(r.get(k) or '') for k in keys) for r in ped_rows)

    return '\n'.join(formatted_ped_rows)

def get_cram_map(dataset):
    reads_map = aapi.get_sample_reads_map_for_seqr(project=dataset)
    reads_list = [l for l in list(set(reads_map.split("\n")))]

    # temporarily
    d = '/Users/michael.franklin/source/sample-metadata/cram-map'
    with open(os.path.join(d, dataset + '-cram-map.tsv'), 'w+') as f:
        f.writelines(reads_list)



def get_token():
    import google.auth.exceptions
    import google.auth.transport.requests

    credential_filename = '/Users/michael.franklin/source/sample-metadata/scripts/seqr-308602-6b221bc0893c.json'
    with open(credential_filename, 'r') as f:
        from google.oauth2 import service_account

        info = json.load(f)
        credentials_content = (info.get('type') == 'service_account') and info or None
        credentials = service_account.IDTokenCredentials.from_service_account_info(
            credentials_content, target_audience=SEQR_AUDIENCE
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token


def sync_all_datasets(ignore: set[str] = None):
    seqr_projects = ProjectApi().get_seqr_projects()
    error_projects = []
    for project in seqr_projects:
        project_name = project['name']
        if ignore and project_name in ignore:
            print(f'Skipping {project_name}')
            continue
        seqr_guid = project.get('meta', {}).get('seqr_guid')
        if not seqr_guid:
            print(f'Skipping "{project_name}" as meta.seqr_guid is not set')
            continue

        try:
            sync_dataset(project_name, seqr_guid)
        except Exception as e:
            error_projects.append((project_name, e))

    if error_projects:
        print(
            'Some projects failed with errors: '
            + '\n'.join(str(o) for o in error_projects)
        )

    return error_projects


def sync_single_dataset_from_name(dataset):
    seqr_projects = ProjectApi().get_seqr_projects()
    for project in seqr_projects:
        project_name = project['name']
        if project_name != dataset:
            continue
        seqr_guid = project.get('meta', {}).get('seqr_guid')
        if not seqr_guid:
            raise ValueError(
                f'{project_name} does NOT have a meta.seqr_guid is not set'
            )

        return sync_dataset(project_name, seqr_guid)

    raise ValueError(f'Could not find {dataset} seqr project')


if __name__ == '__main__':
    # sync_single_dataset_from_name('ohmr4-epilepsy')
    sync_dataset('hereditary-neuro', 'R0013_hereditary_neuro_test')
    # ignore = {
    #     'heartkids',
    #     'acute-care',
    #     'perth-neuro',
    #     'ravenscroft-rdstudy',
    #     'circa',
    #     'hereditary-neuro',
    # }
    # ignore = {'ohmr4-epilepsy'}
    ignore = None
    sync_all_datasets(ignore=ignore)
