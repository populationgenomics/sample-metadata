from typing import Optional
from models.base import SMBase


class ProjectRow(SMBase):
    """Row for project in 'project' table"""

    id: Optional[int] = None
    name: Optional[str] = None
    gcp_id: Optional[str] = None
    dataset: Optional[str] = None
    read_group_name: Optional[str] = None
    write_group_name: Optional[str] = None

    @staticmethod
    def from_db(kwargs):
        """From DB row, with db keys"""
        kwargs = dict(kwargs)
        return ProjectRow(**kwargs)
