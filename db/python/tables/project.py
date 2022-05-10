# pylint: disable=global-statement
import asyncio
from typing import Dict, List, Set, Iterable, Optional, Tuple

import os
from datetime import datetime, timedelta

from databases import Database

from db.python.utils import ProjectId, Forbidden, NoProjectAccess, get_logger
from models.models.project import ProjectRow

from cpg_utils.permissions import get_group_members

# minutes
PROJECT_CACHE_LENGTH = 1
PERMISSIONS_CACHE_LENGTH = 1
_ALLOW_FULL_ACCESS = os.getenv('SM_ALLOWALLACCESS', 'n').lower() in ('y', 'true', '1')
PROJECT_CREATORS_GROUP = os.getenv('SM_PROJECT_CREATORS_GROUP')


def is_full_access():
    """Does SM have full access"""
    return _ALLOW_FULL_ACCESS


def set_full_access(access):
    """Set full_access for future use"""
    global _ALLOW_FULL_ACCESS
    _ALLOW_FULL_ACCESS = access


logger = get_logger()


class ProjectPermissionCacheObject:
    """
    Project permissions object to cache,
    with expiry and 'is_valid' value
    """

    def __init__(self, users: Set[str], expiry=None):
        self.users = users
        self.expiry = expiry or (
            datetime.utcnow() + timedelta(minutes=PERMISSIONS_CACHE_LENGTH)
        )

    def is_valid(self):
        """Is the value still valid, or has it expired"""
        return datetime.utcnow() < self.expiry


class ProjectPermissionsTable:
    """
    Capture project operations and queries
    """

    table_name = 'project'
    _cached_client = None

    _cache_expiry = None
    _cached_project_names: Dict[str, ProjectId] = {}
    _cached_project_by_id: Dict[ProjectId, ProjectRow] = {}

    _cached_permissions: Dict[Tuple[ProjectId, bool], ProjectPermissionCacheObject] = {}

    def __init__(self, connection: Database, allow_full_access=None):

        self.connection: Database = connection
        self.allow_full_access = (
            allow_full_access if allow_full_access is not None else is_full_access()
        )

    async def check_access_to_project_ids(
        self,
        user: str,
        project_ids: Iterable[ProjectId],
        readonly: bool,
        raise_exception=True,
    ) -> bool:
        """Check user has access to list of project_ids"""
        if not project_ids:
            raise Forbidden(
                "You don't have access to this resources, as the resource you requested didn't belong to a project"
            )
        if self.allow_full_access:
            return True
        missing_project_ids = []
        spids = set(project_ids)
        for project_id in spids:
            has_access = await self.check_access_to_project_id(
                user, project_id, readonly=readonly, raise_exception=False
            )
            if not has_access:
                missing_project_ids.append(project_id)

        if missing_project_ids:
            if raise_exception:
                project_map = await self.get_project_id_map()
                missing_project_names = [
                    project_map[pid].name for pid in missing_project_ids
                ]
                raise NoProjectAccess(
                    missing_project_names, readonly=readonly, author=user
                )
            return False

        return True

    async def check_access_to_project_id(
        self, user: str, project_id: ProjectId, readonly: bool, raise_exception=True
    ) -> bool:
        """Check whether a user has access to project_id"""
        if self.allow_full_access:
            return True
        if not readonly:
            # validate write privileges here connection
            pass
        try:
            users = await self.get_allowed_users_for_project_id(
                project_id, readonly=readonly
            )
        except Exception as e:  # pylint: disable=broad-except
            if raise_exception:
                raise e

            return False

        has_access = users is None or user in users
        if not has_access and raise_exception:
            project_name = (await self.get_project_id_map())[project_id].name
            raise NoProjectAccess([project_name], readonly=readonly, author=user)
        return has_access

    async def get_allowed_users_for_project_id(
        self, project_id, readonly: bool
    ) -> Optional[Set[str]]:
        """Get allowed users for a project_id"""
        cache_key = (project_id, readonly)
        if (
            cache_key not in self._cached_permissions
            or not self._cached_permissions[cache_key].is_valid()
        ):
            project_id_map = await self.get_project_id_map()
            project = project_id_map[project_id]
            group_name = (
                project.read_group_name if readonly else project.write_group_name
            )
            if group_name is None:
                project_name = (await self.get_project_id_map())[project_id].name
                read_or_write = 'read' if readonly else 'write'
                raise Exception(
                    f'An internal error occurred when validating access to {project_name}, '
                    f'there must be a value in the DB for "{read_or_write}_group_name" to lookup'
                )

            try:
                start = datetime.utcnow()
                response = get_group_members(group_name)
                logger.debug(
                    f'Took {(datetime.utcnow() - start).total_seconds():.2f} seconds to check group access for {group_name}'
                )

            except Exception as e:
                raise Exception(
                    f'An error occurred when determining access to this project: {e}'
                ) from e

            users = set(response.split(','))
            self._cached_permissions[cache_key] = ProjectPermissionCacheObject(
                users=users
            )

        return self._cached_permissions[cache_key].users

    async def ensure_project_id_cache_is_filled(self):
        """(CACHED) Get map of project names to project IDs"""
        if (
            not ProjectPermissionsTable._cached_project_names
            or ProjectPermissionsTable._cache_expiry < datetime.utcnow()
        ):
            project_rows = await self.get_project_rows(check_permissions=False)
            ProjectPermissionsTable._cached_project_by_id = {
                p.id: p for p in project_rows
            }
            ProjectPermissionsTable._cached_project_names = {
                p.name: p.id for p in project_rows
            }
            ProjectPermissionsTable._cache_expiry = datetime.utcnow() + timedelta(
                minutes=1
            )

    async def get_project_id_map(self) -> Dict[int, ProjectRow]:
        """Get {project_id: ProjectRow} map"""
        await self.ensure_project_id_cache_is_filled()
        return ProjectPermissionsTable._cached_project_by_id

    async def get_project_name_map(self) -> Dict[str, int]:
        """Get {project_name: project_id} map"""
        await self.ensure_project_id_cache_is_filled()
        return ProjectPermissionsTable._cached_project_names

    async def get_project_id_from_name_and_user(
        self, user: str, project_name: str, readonly: bool
    ) -> ProjectId:
        """
        Get projectId from project name and user (email address)
        Returns:
            - int: if user has access to specific project[SampleSequencing.from_db(s) for s in sequence_dicts]
            - None: if user has <no-project> access
            - False if unable to access the specified project
        """
        project_ids = await self.get_project_ids_from_names_and_user(
            user, [project_name], readonly=readonly
        )
        return project_ids[0]

    async def get_project_ids_from_names_and_user(
        self, user: str, project_names: List[str], readonly: bool
    ) -> List[ProjectId]:
        """Get project ids from project names and the user"""
        if not user:
            raise Exception('An internal error occurred during authorization')

        project_name_map = await self.get_project_name_map()
        project_ids = []
        invalid_project_names = []
        for project_name in project_names:
            project_id = project_name_map.get(project_name)
            if not project_id:
                invalid_project_names.append(project_name)
                continue

            can_use = await self.check_access_to_project_id(
                user, project_id, readonly=readonly
            )
            if not can_use:
                invalid_project_names.append(project_name)
                continue

            project_ids.append(project_id)

        if invalid_project_names:
            raise NoProjectAccess(invalid_project_names, readonly=readonly, author=user)

        if len(project_ids) != len(project_names):
            raise Exception(
                'An internal error occurred when mapping project names to IDs'
            )

        return project_ids

    async def check_project_creator_permissions(self, author):
        """Check author has project_creator permissions"""
        # check permissions in here
        if self.allow_full_access:
            return True

        assert PROJECT_CREATORS_GROUP
        members = get_group_members(PROJECT_CREATORS_GROUP)
        if author.lower() not in members:
            raise Forbidden(f'{author} does not have access to creating project')

        return True

    async def get_project_rows(
        self, author: Optional[str] = None, check_permissions=True
    ) -> List[ProjectRow]:
        """Get {name: id} project map"""
        if check_permissions:
            await self.check_project_creator_permissions(author)

        _query = 'SELECT id, name, gcp_id, dataset, read_secret_name, write_secret_name FROM project'
        rows = await self.connection.fetch_all(_query)
        return list(map(ProjectRow.from_db, rows))

    async def get_projects_accessible_by_user(self, author: str, readonly=True):
        """
        Get projects that are accessible by the specified user
        """
        assert author

        _query = 'SELECT id, name FROM project'
        project_id_map = {p[0]: p[1] for p in await self.connection.fetch_all(_query)}

        promises = [
            self.check_access_to_project_id(
                author, pid, readonly=readonly, raise_exception=False
            )
            for pid in project_id_map.keys()
        ]
        has_access_to_project = await asyncio.gather(*promises)
        relevant_project_names = [
            name
            for name, has_access in zip(project_id_map.values(), has_access_to_project)
            if has_access
        ]

        return relevant_project_names

    async def create_project(
        self,
        project_name: str,
        dataset_name: str,
        gcp_project_id: str,
        author: str,
        read_secret_name: str,
        write_secret_name: str,
        create_test_project: bool,
        check_permissions=True,
    ):
        """Create project row"""
        if check_permissions:
            await self.check_project_creator_permissions(author)

        _query = """\
INSERT INTO project (name, gcp_id, dataset, author, read_secret_name, write_secret_name)
VALUES (:name, :gcp_id, :dataset, :author, :read_secret_name, :write_secret_name)
RETURNING ID"""
        values = {
            'name': project_name,
            'dataset': dataset_name,
            'gcp_id': gcp_project_id,
            'author': author,
            'read_secret_name': read_secret_name,
            'write_secret_name': write_secret_name,
        }

        project_id = await self.connection.fetch_val(_query, values)

        if create_test_project:
            values = {
                'name': project_name + '-test',
                'dataset': dataset_name,
                'gcp_id': gcp_project_id,
                'author': author,
                'read_secret_name': read_secret_name.replace('-main-', '-test-'),
                'write_secret_name': write_secret_name.replace('-main-', '-test-'),
            }

            project_id = await self.connection.fetch_val(_query, values)

        return project_id
