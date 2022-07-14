""" This script cleans up the sequence data and metadata within GCS and Metamist """

import logging
import coloredlogs
import click

from cloudpathlib import CloudPath
from google.cloud import storage

from sample_metadata.apis import SampleApi, SequenceApi
from sample_metadata.models import (
    SequenceUpdateModel,
)


client = storage.Client()
sapi = SampleApi()
seqapi = SequenceApi()
coloredlogs.install()


def clean_up_metamist(
    sequence_ids_internal: list[str],
    clean_up_location: str,
):
    """Deleting sequencing entires for specified sequences in metamist
    This will delete all of the reads for the latest sequence."""
    all_locations: list[CloudPath] = []
    sequence_ids: list[int] = []
    for sequence_id in sequence_ids_internal:
        current_sequence = seqapi.get_sequence_by_id(sequence_id)
        reads = current_sequence['meta']['reads']
        current_read_locations: list[CloudPath] = []
        # Skip delete operations if there are no reads for the given sequence
        delete_reads = bool(reads)
        for read in reads:
            for strand in read:
                cloud_path = strand['location']
                read_blob = CloudPath(cloud_path)
                blob_location = read_blob.anchor + read_blob.drive
                # Validates the file that we want to delete, is in the bucket specified.
                if read_blob.exists() and blob_location == clean_up_location:
                    current_read_locations.append(read_blob)
                else:
                    logging.error(
                        f'{read_blob.name} was not found in {clean_up_location}. Skipping delete of {sequence_id} from metamist'
                    )
                    # Skip delete operations, something might be wrong with the reads meta locations.
                    delete_reads = False
                    continue
        # Clear reads metadata, including location, basename, class, checksum & size
        if delete_reads:
            # Delete the reads meta from metamist
            all_locations.extend(current_read_locations)
            sequence_update_model = SequenceUpdateModel(meta={'reads': []})
            api_response = seqapi.update_sequence(sequence_id, sequence_update_model)
            sequence_ids.append(int(api_response))
            logging.info(
                f'{int(api_response)}\'s reads metadata was cleared from metamist'
            )

    return sequence_ids, all_locations


def clean_up_cloud_storage(locations: list[CloudPath]):
    """Given a list of locations of files to be deleted"""
    for location in locations:
        location.unlink()
        logging.info(f'{location.name} was deleted from cloud storage.')


def get_sequences_from_sample(sample_ids: list[str], sequence_type):
    """Return latest sequence ids for sample ids"""
    sequences: list[str] = []
    # Get Sequence Ids From Sample Ids
    sample_sequence_map = seqapi.get_all_sequences_by_sample_ids(sample_ids)
    for sample in sample_sequence_map:
        seq = sample_sequence_map[sample][sequence_type]
        sequences.append(seq)
    return sequences


# TODO: Clean this up, add some helpful messages.
@click.command()
@click.option('--sequence-ids', default=None, multiple=True)
@click.option('--clean-up-location')
@click.option('--project')
@click.option('--external-sample-ids', default=None, multiple=True)
def main(sequence_ids: list[str], clean_up_location, project, external_sample_ids):
    """Clean up"""
    # TODO: Can we take in a batch instead? Or something easier than specifying ID by ID?
    external_sample_ids = list(external_sample_ids)
    sequence_ids = list(sequence_ids)

    if sequence_ids is None and external_sample_ids is None:
        logging.error(
            f'Please specify either sequence-ids or external-sample-ids to be deleted.'
        )
        return

    if external_sample_ids:
        # Get Sample Id Map By External
        sample_map = sapi.get_sample_id_map_by_external(
            project, external_sample_ids, allow_missing=False
        )
        internal_sample_ids = list(sample_map.values())
        # TODO: Pull that sequence type into an arg
        sequence_ids = get_sequences_from_sample(internal_sample_ids, 'genome')

    cleared_ids, locations_to_be_cleared = clean_up_metamist(
        sequence_ids, clean_up_location
    )
    clean_up_cloud_storage(locations_to_be_cleared)

    failed_ids = list(set(sequence_ids) - set(cleared_ids))
    if len(failed_ids) != 0:
        logging.warning(f'Some sequences were not deleted appropriately {failed_ids}')


if __name__ == '__main__':
    main()  # pylint: disable=E1120
