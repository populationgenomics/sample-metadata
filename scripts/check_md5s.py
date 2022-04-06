import os
from typing import Set, Sequence

import click
import hailtop.batch as hb
from google.cloud import storage
from cpg_utils.hail import remote_tmpdir

DRIVER_IMAGE = 'australia-southeast1-docker.pkg.dev/analysis-runner/images/driver:8cc869505251c8396fefef01c42225a7b7930a97-hail-0.2.73.devc6f6f09cec08'


def validate_all_objects_in_directory(gs_dir, filenames: Sequence[str]):
    """Validate files with MD5s in the provided gs directory"""
    backend = hb.ServiceBackend(
        billing_project=os.getenv('HAIL_BILLING_PROJECT'),
        remote_tmpdir=remote_tmpdir(),
    )
    b = hb.Batch('validate_md5s', backend=backend)
    client = storage.Client()

    if not gs_dir.startswith('gs://'):
        raise ValueError(f'Expected GS directory, got: {gs_dir}')

    bucket_name, *components = gs_dir[5:].split('/')

    _blobs = list(client.list_blobs(bucket_name, prefix='/'.join(components)))
    blob_paths = [blob.name for blob in _blobs]
    if filenames:
        sfilenames = set(filenames)
        blob_paths = [
            bname
            for bname in blob_paths
            if os.path.basename(bname) in sfilenames or bname.endswith('.md5')
        ]

    if len(blob_paths) == 0:
        raise ValueError('No files to check MD5s for')

    files: Set[str] = {f'gs://{bucket_name}/{bname}' for bname in blob_paths}
    for obj in files:
        if obj.endswith('.md5'):
            continue
        if f'{obj}.md5' not in files:
            continue

        job = b.new_job(f'validate_{os.path.basename(obj)}')
        job.image(DRIVER_IMAGE)
        validate_md5(job, obj)

    b.run(wait=False)


def validate_md5(job: hb.batch.job, file, md5_path=None) -> hb.batch.job:
    """
    This quickly validates a file and it's md5
    """

    # Calculate md5 checksum.
    md5 = md5_path or f'{file}.md5'
    job.env('GOOGLE_APPLICATION_CREDENTIALS', '/gsa-key/key.json')
    job.command(
        f"""\
set -euxo pipefail
gcloud -q auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
for i in 1 2 3; do {{ gsutil cat {file} | md5sum | cut -d " " -f1 > /tmp/uploaded.md5 }} && break || {{ echo "Sleeping" && sleep 20 }}; done
gsutil cat {file} | md5sum | cut -d " " -f1 > /tmp/uploaded.md5
diff <(cat /tmp/uploaded.md5) <(gsutil cat {md5} | cut -d " " -f1)
    """
    )

    return job


@click.command()
@click.argument('gs_dir')
@click.option(
    '--filename',
    multiple=True,
    help='Specifying the parameter will limit to a specific set of filenames',
)
def main(gs_dir, filename):
    """Main from CLI"""
    validate_all_objects_in_directory(gs_dir=gs_dir, filenames=filename)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
