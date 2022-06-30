import csv
import io
from datetime import date
from typing import List, Optional, Dict, Any

from fastapi import APIRouter
from fastapi.params import Body, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from api.utils.db import (
    get_projectless_db_connection,
    get_project_readonly_connection,
    get_project_write_connection,
    Connection,
)
from api.utils.export import ExportType
from db.python.layers.analysis import AnalysisLayer
from db.python.tables.project import ProjectPermissionsTable
from models.enums import AnalysisType, AnalysisStatus, SequenceType
from models.models.analysis import Analysis
from models.models.sample import (
    sample_id_transform_to_raw_list,
    sample_id_format_list,
)

router = APIRouter(prefix='/analysis', tags=['analysis'])


class AnalysisModel(BaseModel):
    """
    Model for 'createNewAnalysis'
    """

    sample_ids: List[str]
    type: AnalysisType
    status: AnalysisStatus
    meta: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    active: bool = True
    # please don't use this, unless you're the analysis-runner,
    # the usage is tracked ... (Ծ_Ծ)
    author: Optional[str] = None


class AnalysisUpdateModel(BaseModel):
    """Update analysis model"""

    status: AnalysisStatus
    output: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    active: Optional[bool] = None


class AnalysisQueryModel(BaseModel):
    """Used to query for many analysis"""

    # sample_ids means it contains the analysis contains at least one of the sample_ids in the list
    sample_ids: Optional[List[str]]
    # # sample_ids_all means the analysis contains ALL of the sample_ids
    # sample_ids_all: List[str] = None
    projects: List[str]
    type: Optional[AnalysisType] = None
    status: Optional[AnalysisStatus] = None
    meta: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    active: Optional[bool] = None


@router.put('/{project}/', operation_id='createNewAnalysis', response_model=int)
async def create_new_analysis(
    analysis: AnalysisModel, connection: Connection = get_project_write_connection
):
    """Create a new analysis"""

    atable = AnalysisLayer(connection)

    sample_ids = sample_id_transform_to_raw_list(analysis.sample_ids)

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
    assert connection.project
    result = await atable.get_all_sample_ids_without_analysis_type(
        connection.project, analysis_type
    )
    return {'sample_ids': sample_id_format_list(result)}


@router.get(
    '/{project}/incomplete',
    operation_id='getIncompleteAnalyses',
)
async def get_incomplete_analyses(
    connection: Connection = get_project_readonly_connection,
):
    """Get analyses with status queued or in-progress"""
    atable = AnalysisLayer(connection)
    assert connection.project
    results = await atable.get_incomplete_analyses(project=connection.project)
    if results:
        for result in results:
            if result.sample_ids:
                result.sample_ids = list(sample_id_format_list(result.sample_ids))

    return results


@router.get(
    '/{project}/{analysis_type}/latest-complete',
    operation_id='getLatestCompleteAnalysisForType',
)
async def get_latest_complete_analysis_for_type(
    analysis_type: AnalysisType,
    connection: Connection = get_project_readonly_connection,
):
    """Get (SINGLE) latest complete analysis for some analysis type"""
    alayer = AnalysisLayer(connection)
    assert connection.project
    analysis = await alayer.get_latest_complete_analysis_for_type(
        project=connection.project, analysis_type=analysis_type
    )
    if analysis and analysis.sample_ids:
        analysis.sample_ids = list(sample_id_format_list(analysis.sample_ids))

    return analysis


@router.post(
    '/{project}/{analysis_type}/latest-complete',
    operation_id='getLatestCompleteAnalysisForTypePost',
)
async def get_latest_complete_analysis_for_type_post(
    analysis_type: AnalysisType,
    meta: Dict[str, Any] = Body(..., embed=True),  # type: ignore[assignment]
    connection: Connection = get_project_readonly_connection,
):
    """
    Get SINGLE latest complete analysis for some analysis type
    (you can specify meta attributes in this route)
    """
    alayer = AnalysisLayer(connection)
    assert connection.project
    analysis = await alayer.get_latest_complete_analysis_for_type(
        project=connection.project,
        analysis_type=analysis_type,
        meta=meta,
    )
    if analysis and analysis.sample_ids:
        analysis.sample_ids = list(sample_id_format_list(analysis.sample_ids))

    return analysis


@router.post(
    '/{project}/{analysis_type}/latest-complete/for-samples',
    operation_id='getLatestAnalysisForSamplesAndType',
)
async def get_latest_analysis_for_samples_and_type(
    sample_ids: List[str],
    analysis_type: AnalysisType,
    allow_missing: bool = True,
    connection: Connection = get_project_readonly_connection,
):
    """Get latest complete gvcfs for sample"""
    atable = AnalysisLayer(connection)
    results = await atable.get_latest_complete_analysis_for_samples_and_type(
        analysis_type=analysis_type,
        sample_ids=sample_id_transform_to_raw_list(sample_ids),
        allow_missing=allow_missing,
    )

    if results:
        for result in results:
            result.sample_ids = sample_id_format_list(result.sample_ids)

    return results


@router.get(
    '/{analysis_id}/details',
    operation_id='getAnalysisById',
    # mfranklin: uncomment when we supported OpenAPI 3.1
    # response_model=Analysis
)
async def get_analysis_by_id(
    analysis_id: int, connection: Connection = get_projectless_db_connection
):
    """Get analysis by ID"""
    atable = AnalysisLayer(connection)
    result = await atable.get_analysis_by_id(analysis_id)
    result.sample_ids = sample_id_format_list(result.sample_ids)
    return result


@router.post('/query', operation_id='queryAnalyses')
async def query_analyses(
    query: AnalysisQueryModel, connection: Connection = get_projectless_db_connection
):
    """Get analyses by some criteria"""
    if not query.projects:
        raise ValueError('Must specify "projects"')

    pt = ProjectPermissionsTable(connection=connection.connection)
    project_ids = await pt.get_project_ids_from_names_and_user(
        connection.author, query.projects, readonly=True
    )
    atable = AnalysisLayer(connection)
    analyses = await atable.query_analysis(
        sample_ids=sample_id_transform_to_raw_list(query.sample_ids)
        if query.sample_ids
        else None,
        project_ids=project_ids,
        analysis_type=query.type,
        status=query.status,
        meta=query.meta,
        output=query.output,
        active=query.active,
    )
    for analysis in analyses:
        analysis.sample_ids = sample_id_format_list(analysis.sample_ids)

    return analyses


@router.get('/analysis-runner', operation_id='getAnalysisRunnerLog')
async def get_analysis_runner_log(
    project_names: List[str] = None,
    connection: Connection = get_projectless_db_connection,
) -> List[Analysis]:
    """
    Get log for the analysis-runner, useful for checking this history of analysis
    """
    atable = AnalysisLayer(connection)
    project_ids = None
    if project_names:
        pt = ProjectPermissionsTable(connection=connection.connection)
        project_ids = await pt.get_project_ids_from_names_and_user(
            connection.author, project_names, readonly=True
        )

    results = await atable.get_analysis_runner_log(project_ids=project_ids)
    return results


@router.get(
    '/{project}/sample-cram-path-map',
    operation_id='getSamplesReadsMap',
    tags=['seqr'],
)
async def get_sample_reads_map(
    export_type: ExportType = ExportType.JSON,
    sequence_types: List[SequenceType] = Query(...),
    connection: Connection = get_project_readonly_connection,
):
    """
    Get map of ExternalSampleId  pathToCram  InternalSampleID for seqr

    Description:
        Column 1: Individual ID
        Column 2: gs:// Google bucket path or server filesystem path for this Individual
        Column 3: Sample ID for this file, if different from the Individual ID.
                    Used primarily for gCNV files to identify the sample in the batch path
    """

    at = AnalysisLayer(connection)
    assert connection.project
    objs = await at.get_sample_cram_path_map_for_seqr(
        project=connection.project, sequence_types=sequence_types
    )

    if export_type == ExportType.JSON:
        return objs

    keys = ['participant_id', 'output', 'sample_id']
    rows = [[r[k] for k in keys] for r in objs]

    output = io.StringIO()
    writer = csv.writer(output, delimiter=export_type.get_delimiter())
    writer.writerows(rows)

    basefn = f'{connection.project}-seqr-igv-paths-{date.today().isoformat()}'

    return StreamingResponse(
        iter(output.getvalue()),
        media_type=export_type.get_mime_type(),
        headers={
            'Content-Disposition': f'filename={basefn}{export_type.get_extension()}'
        },
    )
