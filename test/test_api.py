import os
import unittest
import uuid
from typing import Dict, List

from sample_metadata import (
    NewSequence,
    AnalysisModel,
    NewSample,
    SampleApi,
    AnalysisApi,
    SequenceApi,
    SampleUpdateModel, AnalysisUpdateModel, SequenceUpdateModel,
)

sapi = SampleApi()
aapi = AnalysisApi()
seqapi = SequenceApi()

PROJECT = os.environ.get('PROJECT', 'dev')
ANALYSIS_PROJECT = os.environ.get('ANALYSIS_PROJECT', PROJECT)


class AddSampleTestCase(unittest.TestCase):
    """
    Test adding sample. All other test in this module will add samples in setUp, so
    they assume adding sample is tested here.
    """
    def test_add_sample(self):
        """
        Test adding sample.
        """
        extid, cpgid = _add_sample()
        print(f'{self.__class__.__name__}: added sample {extid}/{cpgid}')
        samples_by_cpgid = _get_samples_by_cpgid()
        samples_by_extid = [s['external_id'] for s in samples_by_cpgid.values()]
        self.assertIn(cpgid, samples_by_cpgid)
        self.assertIn(extid, samples_by_extid)
        sapi.update_sample(cpgid, SampleUpdateModel(active=False))


class AddAnalysisSequencingTestCase(unittest.TestCase):
    """
    Assuming adding samples is working, test basic update commands:
    add analysis and sequences on a sample, test update sample.
    """
    cpgid = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Add sample that will be used as a base for tests.
        """
        extid, cpgid = _add_sample()
        print(f'{cls.__name__}: added sample {extid}/{cpgid}')
        cls.cpgid = cpgid

    def test_update_sample(self):
        """
        Test updating sample meta.
        """
        sample = _get_samples_by_cpgid()[self.cpgid]
        self.assertEqual(sample['meta']['key'], 'initial_value')
        sapi.update_sample(self.cpgid, SampleUpdateModel(
            meta={'key': 'updated_value'}
        ))
        sample = _get_samples_by_cpgid()[self.cpgid]
        self.assertEqual(sample['meta']['key'], 'updated_value')

    def test_add_sequence(self):
        """
        Test adding sequence entry.
        """
        _add_sequenece(self.cpgid)
        seq_by_cpgid = _get_sequence_by_cpgid([self.cpgid])
        self.assertIn(self.cpgid, seq_by_cpgid)

    def test_queue_single_sample_analysis(self):
        """
        Test adding analysis.
        """
        added_aid = _queue_analysis(self.cpgid)
        analyses = aapi.get_incomplete_analyses(project=ANALYSIS_PROJECT)
        self.assertGreater(len(analyses), 0)
        self.assertIn(added_aid, (a['id'] for a in analyses))
        analysis = next(a for a in analyses if a['id'] == added_aid)
        self.assertEqual(len(analysis['sample_ids']), 1)
        self.assertListEqual(analysis['sample_ids'], [self.cpgid])

    def test_add_joint_calling_analysis(self):
        """
        Test adding analysis for multiple samples.
        """
        added_aid = aapi.create_new_analysis(
            project=ANALYSIS_PROJECT,
            analysis_model=AnalysisModel(
                sample_ids=[self.cpgid],
                type='joint-calling',
                output='joint-called.vcf',
                status='completed',
            )
        )
        analysis = aapi.get_latest_complete_analysis_for_type(
            project=ANALYSIS_PROJECT, analysis_type='joint-calling'
        )
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['id'], added_aid)
        self.assertListEqual(analysis['sample_ids'], [self.cpgid])

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Delete the sample.
        """
        sapi.update_sample(cls.cpgid, SampleUpdateModel(active=False))


class UpdateTestCase(unittest.TestCase):
    """
    Assuming adding samples, analyses and sequences is working,
    test updating analyses and sequences
    """
    cpgid = None
    analysis_id = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Add sample, analysis and sample for further tests.
        """
        extid, cpgid = _add_sample()
        print(f'{cls.__name__}: added sample {extid}/{cpgid}')
        cls.cpgid = cpgid
        cls.analysis_id = _queue_analysis(cls.cpgid)
        _add_sequenece(cls.cpgid)

    def test_update_analysis_to_completed(self):
        """
        Test updating analysis state to "completed".
        """
        aapi.update_analysis_status(
            analysis_id=self.analysis_id,
            analysis_update_model=AnalysisUpdateModel(status='completed'),
        )
        analyses = aapi.get_latest_analysis_for_samples_and_type(
            project=ANALYSIS_PROJECT,
            analysis_type='cram',
            request_body=[self.cpgid],
        )
        self.assertGreater(len(analyses), 0)
        self.assertIn(self.analysis_id, (a['id'] for a in analyses))
        analysis = next(a for a in analyses if a['id'] == self.analysis_id)
        self.assertEqual(len(analysis['sample_ids']), 1)
        self.assertListEqual(analysis['sample_ids'], [self.cpgid])

    def test_update_sequence_status(self):
        """
        Test updating "sequence" status.
        """
        seq_by_cpgid = _get_sequence_by_cpgid([self.cpgid])
        self.assertIn(self.cpgid, seq_by_cpgid)
        seq = seq_by_cpgid[self.cpgid]
        self.assertEqual(seq['status'], 'received')

        seqapi.update_sequence(seq['id'], SequenceUpdateModel(status='uploaded'))

        seq_by_cpgid = _get_sequence_by_cpgid([self.cpgid])
        seq = seq_by_cpgid[self.cpgid]
        self.assertEqual(seq['status'], 'uploaded')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Delete the sample.
        """
        sapi.update_sample(cls.cpgid, SampleUpdateModel(active=False))


class DeactivateSampleTestCase(unittest.TestCase):
    """
    Assuming adding samples is working, test "removing" samples by unsetting "active"
    """
    @classmethod
    def setUpClass(cls) -> None:
        """
        Add sample for further tests.
        """
        extid, cpgid = _add_sample()
        print(f'{cls.__name__}: added sample {extid}/{cpgid}')
        cls.cpgid = cpgid

    def deactivate_sample(self):
        """
        Test "deactivating" a sample.
        """
        sapi.update_sample(self.cpgid, SampleUpdateModel(active=False))
        self.assertNotIn(self.cpgid, _get_samples_by_cpgid())


def _get_samples_by_cpgid() -> Dict[str, Dict]:
    samples = sapi.get_samples(
        body_get_samples_by_criteria_api_v1_sample_post={
            'project_ids': [PROJECT],
            'active': True,
        }
    )
    return {s['id']: s for s in samples}


def _add_sample():
    samples_by_cpgid = _get_samples_by_cpgid()
    samples_by_extid = [s['external_id'] for s in samples_by_cpgid.values()]
    while (extid := f'TEST_{str(uuid.uuid4())[:6]}') in samples_by_extid:
        pass
    cpgid = sapi.create_new_sample(
        PROJECT,
        NewSample(
            external_id=extid,
            type='blood',
            meta={'key': 'initial_value'}
        ),
    )
    return extid, cpgid


def _get_sequence_by_cpgid(cpgids: List[str]) -> Dict[str, Dict]:
    seqs = seqapi.get_sequences_by_sample_ids(cpgids)
    return {s['sample_id']: s for s in seqs}


def _queue_analysis(cpgid):
    return aapi.create_new_analysis(
        project=ANALYSIS_PROJECT,
        analysis_model=AnalysisModel(
            type='cram',
            output=f'result.cram',
            status='queued',
            sample_ids=[cpgid],
        )
    )


def _add_sequenece(cpgid):
    meta = {
        'reads': [
            [
                'NA12878_L002_R1.fq',
                'NA12878_L002_R1.fq',
            ],
            [
                'NA12878_L001_R2.fq',
                'NA12878_L002_R2.fq',
            ],
        ],
        'read_type': 'fastq',
    }
    return seqapi.create_new_sequence(NewSequence(
        sample_id=cpgid,
        meta=meta,
        type='wgs',
        status='received',
    ))
