from typing import Optional, Dict, Any
import json
from pydantic import BaseModel


class ParticipantModel(BaseModel):
    """Update participant model"""

    id: int
    external_id: str = None
    reported_sex: Optional[int] = None
    reported_gender: Optional[str] = None
    karyotype: Optional[str] = None
    meta: Dict

    def __init__(self, **data: Any) -> None:
        if 'meta' in data and isinstance(data['meta'], str):
            data['meta'] = json.loads(data['meta'])

        super().__init__(**data)
