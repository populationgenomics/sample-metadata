from typing import List, Optional, Dict, Any

import io
import csv
from datetime import date

from fastapi import APIRouter
from fastapi.params import Body
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from models.enums import AnalysisType, AnalysisStatus
from models.models.analysis import Analysis
from models.models.sample import sample_id_transform_to_raw, sample_id_format
from db.python.layers.analysis import AnalysisLayer

from api.utils.db import (
    get_projectless_db_connection,
    get_project_readonly_connection,
    get_project_write_connection,
    Connection,
)

router = APIRouter(prefix='/analysis', tags=['analysis'])


class AnalysisModel(BaseModel):
    """
    Model for 'createNewAnalysis'
    """

    sample_ids: List[str]
    type: AnalysisType
    status: AnalysisStatus
    meta: Dict[str, Any] = None
    output: str = None
    active: bool = True
    # please don't use this, unless you're the analysis-runner,
    # the usage is tracked ... (Ծ_Ծ)
    author: Optional[str] = None


class AnalysisUpdateModel(BaseModel):
    """Update analysis model"""

    status: AnalysisStatus
    output: Optional[str] = None
    meta: Dict[str, Any] = None
    active: bool = None


@router.put('/{project}/', operation_id='createNewAnalysis', response_model=int)
async def create_new_analysis(
    analysis: AnalysisModel, connection: Connection = get_project_write_connection
):
    """Create a new analysis"""

    atable = AnalysisLayer(connection)

    sample_ids = sample_id_transform_to_raw(analysis.sample_ids)

    analysis_id = await atable.insert_analysis(
        analysis_type=analysis.type,
        status=analysis.status,
        sample_ids=sample_ids,
        meta=analysis.meta,
        output=analysis.output,
        # analysis-runner: usage is tracked through `on_behalf_of`
        author=analysis.author,
    )

    return analysis_id


@router.patch('/{analysis_id}/', operation_id='updateAnalysisStatus')
async def update_analysis_status(
    analysis_id: int,
    analysis: AnalysisUpdateModel,
    connection: Connection = get_projectless_db_connection,
):
    """Update status of analysis"""
    atable = AnalysisLayer(connection)
    await atable.update_analysis(
        analysis_id, status=analysis.status, output=analysis.output, meta=analysis.meta
    )
    return True


@router.get(
    '/{project}/type/{analysis_type}/not-included-sample-ids',
    operation_id='getAllSampleIdsWithoutAnalysisType',
)
async def get_all_sample_ids_without_analysis_type(
    analysis_type: AnalysisType,
    connection: Connection = get_project_readonly_connection,
):
    """get_all_sample_ids_without_analysis_type"""
    atable = AnalysisLayer(connection)
    result = await atable.get_all_sample_ids_without_analysis_type(
        connection.project, analysis_type
    )
    return {'sample_ids': sample_id_format(result)}


@router.get(
    '/{project}/incomplete',
    operation_id='getIncompleteAnalyses',
)
async def get_incomplete_analyses(
    connection: Connection = get_project_readonly_connection,
):
    """Get analyses with status queued or in-progress"""
    atable = AnalysisLayer(connection)
    results = await atable.get_incomplete_analyses(project=connection.project)
    if results:
        for result in results:
            result.sample_ids = sample_id_format(result.sample_ids)

    return results


@router.get(
    '/{project}/{analysis_type}/latest-complete',
    operation_id='getLatestCompleteAnalysisForType',
)
async def get_latest_complete_analysis_for_type(
    analysis_type: AnalysisType,
    connection: Connection = get_project_readonly_connection,
):
    """Get latest complete analysis for some analysis type"""
    alayer = AnalysisLayer(connection)
    analysis = await alayer.get_latest_complete_analysis_for_type(
        project=connection.project, analysis_type=analysis_type
    )
    if analysis:
        analysis.sample_ids = sample_id_format(analysis.sample_ids)

    return analysis


@router.post(
    '/{project}/{analysis_type}/latest-complete',
    operation_id='getLatestCompleteAnalysisForTypePost',
)
async def get_latest_complete_analysis_for_type_post(
    analysis_type: AnalysisType,
    meta: Dict[str, Any] = Body(..., embed=True),
    connection: Connection = get_project_readonly_connection,
):
    """Get latest complete analysis for some analysis type"""
    alayer = AnalysisLayer(connection)
    analysis = await alayer.get_latest_complete_analysis_for_type(
        project=connection.project,
        analysis_type=analysis_type,
        meta=meta,
    )
    if analysis:
        analysis.sample_ids = sample_id_format(analysis.sample_ids)

    return analysis


@router.get(
    '/{project}/{analysis_type}/latest-complete',
    operation_id='getLatestCompleteAnalysisForType',
)
async def get_latest_complete_analysis_for_type(
    analysis_type: AnalysisType,
    connection: Connection = get_project_readonly_connection,
):
    """Get latest complete analysis for some analysis type"""
    alayer = AnalysisLayer(connection)
    analysis = await alayer.get_latest_complete_analysis_for_type(
        project=connection.project, analysis_type=analysis_type
    )
    if analysis:
        analysis.sample_ids = sample_id_format(analysis.sample_ids)

    return analysis


@router.get(
    '/{analysis_id}/details', operation_id='getAnalysisById', response_model=Analysis
)
async def get_analysis_by_id(
    analysis_id: int, connection: Connection = get_projectless_db_connection
):
    """Get analysis by ID"""
    atable = AnalysisLayer(connection)
    result = await atable.get_analysis_by_id(analysis_id)
    result.sample_ids = sample_id_format(result.sample_ids)
    return result


@router.get(
    '/{project}/sample-cram-path-map/tsv',
    operation_id='getSampleReadsMapForSeqr',
    response_class=StreamingResponse,
)
async def get_sample_reads_map_for_seqr(
    connection: Connection = get_project_readonly_connection,
):
    """
    Get map of ExternalSampleId  pathToCram  InternalSampleID for seqr

    Description:
        Column 1: Individual ID
        Column 2: gs:// Google bucket path or server filesystem path for this Individual
        Column 3: Sample ID for this file, if different from the Individual ID. Used primarily for gCNV files to identify the sample in the batch path
    """

    at = AnalysisLayer(connection)
    rows = await at.get_sample_cram_path_map_for_seqr(project=connection.project)

    # remap sample ids
    for row in rows:
        row[2] = sample_id_format(row[2])

    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t')
    writer.writerows(rows)

    basefn = f'{connection.project}-seqr-igv-paths-{date.today().isoformat()}'

    return StreamingResponse(
        iter(output.getvalue()),
        media_type='text/tab-separated-values',
        headers={'Content-Disposition': f'filename={basefn}.tsv'},
    )
