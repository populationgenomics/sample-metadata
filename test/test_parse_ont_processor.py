import unittest
from io import StringIO
from unittest.mock import patch

from test.testbase import run_test_as_sync

from scripts.process_ont_products import OntProductParser


class TestOntSampleSheetParser(unittest.TestCase):
    """Test the TestOntSampleSheetParser"""

    @run_test_as_sync
    @patch('sample_metadata.apis.SampleApi.get_sample_id_map_by_external')
    @patch('cpg_utils.hail_batch.get_config')
    @patch('sample_metadata.parser.cloudhelper.AnyPath')
    async def test_single_row_all_files_exist(
        self, mock_anypath, mock_get_config, mock_get_sample_id
    ):
        """
        Test processing one row with all files existing
        """
        mock_get_sample_id.return_value = {'Sample01': 'CPG001'}
        mock_get_config.return_value = {
            'workflow': {'dataset': 'ONT-TEST', 'access_level': 'main'}
        }
        mock_anypath.return_value.stat.return_value.st_size = 111

        rows = [
            'Experiment name,Sample ID,Alignment_file,Alignment_software,SV_file,SV_software,SNV_file,SNV_software,Indel_file,Indel_software',
            'PBX10,Sample01,Sample01.bam,minimap2/2.22,Sample01.sv.vcf.gz,"Sniffles2, Version 2.0.2",Sample01.snvs.vcf.gz,Clair3 v0.1-r7,Sample01.indels.vcf.gz,Clair3 v0.1-r7',
        ]

        parser = OntProductParser(
            search_paths=[],
            # doesn't matter, we're going to mock the call anyway
            project='dev',
            dry_run=True,
        )

        parser.skip_checking_gcs_objects = True
        fs = [
            'Sample01.bam',
            'Sample01.sv.vcf.gz',
            'Sample01.snvs.vcf.gz',
            'Sample01.indels.vcf.gz',
        ]
        parser.filename_map = {k: 'gs://BUCKET/FAKE/' + k for k in fs}
        parser.skip_checking_gcs_objects = True

        file_contents = '\n'.join(rows)
        analyses = await parser.parse_manifest(
            StringIO(file_contents),
            delimiter=',',
        )

        self.assertEqual(4, len(analyses))
