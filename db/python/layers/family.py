# pylint: disable=used-before-assignment
import logging
from typing import List, Union, Optional, Dict

from db.python.connect import Connection
from db.python.layers.base import BaseLayer
from db.python.layers.participant import ParticipantLayer
from db.python.tables.family import FamilyTable
from db.python.tables.family_participant import FamilyParticipantTable
from db.python.tables.participant import ParticipantTable
from db.python.tables.project import ProjectId
from db.python.tables.sample import SampleTable


class PedRow:
    """Class for capturing a row in a pedigree"""

    ALLOWED_SEX_VALUES = [0, 1, 2]
    ALLOWED_AFFECTED_VALUES = [-9, 0, 1, 2]

    PedRowKeys = {
        # seqr individual template:
        # Family ID, Individual ID, Paternal ID, Maternal ID, Sex, Affected, Status, Notes
        'family_id': {'familyid', 'family id', 'family', 'family_id'},
        'individual_id': {'individualid', 'id', 'individual_id', 'individual id'},
        'paternal_id': {'paternal_id', 'paternal id', 'paternalid', 'father'},
        'maternal_id': {'maternal_id', 'maternal id', 'maternalid', 'mother'},
        'sex': {'sex', 'gender'},
        'affected': {'phenotype', 'affected', 'phenotypes', 'affected status'},
        'notes': {'notes'},
    }

    def as_row(self, empty_participant_value=None):
        """Get the row as ordered in the header"""
        return [
            self.family_id,
            self.individual_id,
            self.paternal_id or empty_participant_value,
            self.maternal_id or empty_participant_value,
            self.sex,
            self.affected,
            self.notes or '',
        ]

    @staticmethod
    def default_header():
        """Default header (corresponds to the __init__ keys)"""
        return [
            'family_id',
            'individual_id',
            'paternal_id',
            'maternal_id',
            'sex',
            'affected',
            'notes',
        ]

    @staticmethod
    def row_header():
        """Default RowHeader for output"""
        return [
            '#Family ID',
            'Individual ID',
            'Paternal ID',
            'Maternal ID',
            'Sex',
            'Affected',
        ]

    def __init__(
        self,
        family_id,
        individual_id,
        paternal_id,
        maternal_id,
        sex,
        affected,
        notes=None,
    ):
        self.family_id = family_id.strip()
        self.individual_id = individual_id.strip()
        self.paternal_id = None
        self.maternal_id = None
        self.paternal_id = self.check_linking_id(paternal_id, 'paternal_id')
        self.maternal_id = self.check_linking_id(maternal_id, 'maternal_id')
        self.sex = self.parse_sex(sex)
        self.affected = self.parse_affected_status(affected)
        self.notes = notes

    @staticmethod
    def check_linking_id(linking_id, description: str, blank_values=('0', '')):
        """Check that the ID is a valid value, or return None if it's a blank value"""
        if linking_id is None:
            return None
        if isinstance(linking_id, int):
            linking_id = str(linking_id)

        if isinstance(linking_id, str):
            if linking_id.lower() in blank_values:
                return None
            return linking_id

        raise TypeError(
            f'Unexpected type {type(linking_id)} ({linking_id}) '
            f'for {description}, expected "str"'
        )

    @staticmethod
    def parse_sex(sex: Union[str, int]) -> int:
        """
        Parse the pedigree SEX value:
            0: unknown
            1: male (also accepts 'm')
            2: female (also accepts 'f')
        """
        if isinstance(sex, str) and sex.isdigit():
            sex = int(sex)
        if isinstance(sex, int):
            if sex in PedRow.ALLOWED_SEX_VALUES:
                return sex
            raise ValueError(
                f'Sex value ({sex}) was not an expected value {PedRow.ALLOWED_SEX_VALUES}.'
            )

        sl = sex.lower()
        if sl in ('m', 'male'):
            return 1
        if sl in ('f', 'female'):
            return 2
        if sl == 'sex':
            raise ValueError(
                f'Unknown sex "{sex}", did you mean to call import_pedigree with has_headers=True?'
            )
        raise ValueError(
            f'Unknown sex "{sex}", please ensure sex is in {PedRow.ALLOWED_SEX_VALUES}'
        )

    @staticmethod
    def parse_affected_status(affected):
        """
        Parse the pedigree "AFFECTED" value:
            -9 / 0: unknown
            1: unaffected
            2: affected
        """
        if isinstance(affected, str) and not affected.isdigit():
            affected = affected.lower().strip()
            if affected in ['unknown']:
                return 0
            if affected in ['n', 'no']:
                return 1
            if affected in ['y', 'yes', 'affected']:
                return 2

        affected = int(affected)
        if affected not in PedRow.ALLOWED_AFFECTED_VALUES:
            raise ValueError(
                f'Affected value {affected} was not in expected value: {PedRow.ALLOWED_AFFECTED_VALUES}'
            )

        return affected

    def __str__(self):
        return f'PedRow: {self.individual_id} ({self.sex})'

    @staticmethod
    def order(rows: List) -> List:
        """
        Order a list of PedRows, but also validates:
        - There are no circular dependencies
        - All maternal / paternal IDs are found in the pedigree
        """
        rows_to_order: List['PedRow'] = [*rows]
        ordered = []
        seen_individuals = set()
        remaining_iterations_in_round = len(rows_to_order)

        while len(rows_to_order) > 0:
            row = rows_to_order.pop(0)
            reqs = [row.paternal_id, row.maternal_id]
            if all(r is None or r in seen_individuals for r in reqs):
                remaining_iterations_in_round = len(rows_to_order)
                ordered.append(row)
                seen_individuals.add(row.individual_id)
            else:
                remaining_iterations_in_round -= 1
                rows_to_order.append(row)

            # makes more sense to keep this comparison separate:
            #   - If remaining iterations is or AND we still have rows
            #   - Then raise an Exception
            # pylint: disable=chained-comparison
            if remaining_iterations_in_round <= 0 and len(rows_to_order) > 0:
                participant_ids = ', '.join(r.individual_id for r in rows_to_order)
                raise Exception(
                    "There was an issue in the pedigree, either a parent wasn't found in the pedigree, "
                    "or a circular dependency detected (eg: someone's child is an ancestor's parent). "
                    f"Can't resolve participants: {participant_ids}"
                )

        return ordered

    @staticmethod
    def validate_sexes(rows: List['PedRow'], throws=True) -> bool:
        """
        Validate that individuals listed as mothers and fathers
        have either unknown sex, male if paternal, and female if maternal.

        Future note: The pedigree has a simplified view of sex, especially
        how it relates families together. This function might not handle
        more complex cases around intersex disorders within families. The
        best advice is either to skip this check, or provide sex as 0 (unknown)

        :param throws: If True is provided (default), raise a ValueError, else just return False
        """
        keyed: Dict[str, PedRow] = {r.individual_id: r for r in rows}
        paternal_ids = [r.paternal_id for r in rows if r.paternal_id]
        mismatched_pat_sex = [
            pid for pid in paternal_ids if keyed[pid].sex not in (0, 1)
        ]
        maternal_ids = [r.maternal_id for r in rows if r.maternal_id]
        mismatched_mat_sex = [
            mid for mid in maternal_ids if keyed[mid].sex not in (0, 2)
        ]

        messages = []
        if mismatched_pat_sex:
            actual_values = ', '.join(
                f'{pid} ({keyed[pid].sex})' for pid in mismatched_pat_sex
            )
            messages.append('(0, 1) as they are listed as fathers: ' + actual_values)
        if mismatched_mat_sex:
            actual_values = ', '.join(
                f'{pid} ({keyed[pid].sex})' for pid in mismatched_mat_sex
            )
            messages.append('(0, 2) as they are listed as mothers: ' + actual_values)

        if messages:
            message = 'Expected individuals have sex values:' + ''.join(
                '\n\t' + m for m in messages
            )
            if throws:
                raise ValueError(message)
            logging.warning(message)
            return False

        return True

    @staticmethod
    def parse_header_order(header: List[str]):
        """
        Takes a list of unformatted headers, and returns a list of ordered init_keys

        >>> PedRow.parse_header_order(['family', 'mother', 'paternal id', 'phenotypes', 'gender'])
        ['family_id', 'maternal_id', 'paternal_id', 'affected', 'sex']

        >>> PedRow.parse_header_order(['#family id'])
        ['family_id']

        >>> PedRow.parse_header_order(['unexpected header'])
        Traceback (most recent call last):
        ValueError: Unable to identity header elements: "unexpected header"
        """
        ordered_init_keys = []
        unmatched = []
        for item in header:
            litem = item.lower().strip().strip('#')
            found = False
            for h, options in PedRow.PedRowKeys.items():
                for potential_key in options:
                    if potential_key == litem:
                        ordered_init_keys.append(h)
                        found = True
                        break
                if found:
                    break

            if not found:
                unmatched.append(item)

        if unmatched:
            unmatched_headers_str = ', '.join(f'"{u}"' for u in unmatched)
            raise ValueError(
                'Unable to identity header elements: ' + unmatched_headers_str
            )

        return ordered_init_keys


class FamilyLayer(BaseLayer):
    """Layer for import logic"""

    def __init__(self, connection: Connection):
        super().__init__(connection)
        self.stable = SampleTable(connection)
        self.ftable = FamilyTable(connection)
        self.fptable = FamilyParticipantTable(self.connection)

    async def get_families(
        self,
        project: int = None,
        participant_ids: List[int] = None,
        sample_ids: List[int] = None,
    ):
        """Get all families for a project"""
        project = project or self.connection.project

        # Merge sample_id and participant_ids into a single list
        all_participants = participant_ids if participant_ids else []

        # Find the participants from the given samples
        if sample_ids is not None and len(sample_ids) > 0:
            _, samples = await self.stable.get_samples_by(
                project_ids=[project], sample_ids=sample_ids
            )

            all_participants += [
                int(s.participant_id) for s in samples if s.participant_id
            ]
            all_participants = list(set(all_participants))

        return await self.ftable.get_families(
            project=project, participant_ids=all_participants
        )

    async def update_family(
        self,
        id_: int,
        external_id: str = None,
        description: str = None,
        coded_phenotype: str = None,
        check_project_ids: bool = True,
    ) -> bool:
        """Update fields on some family"""
        if check_project_ids:
            project_ids = await self.ftable.get_projects_by_family_ids([id_])
            await self.ptable.check_access_to_project_ids(
                self.author, project_ids, readonly=False
            )

        return await self.ftable.update_family(
            id_=id_,
            external_id=external_id,
            description=description,
            coded_phenotype=coded_phenotype,
        )

    async def get_pedigree(
        self,
        project: ProjectId,
        family_ids: List[int] = None,
        # pylint: disable=invalid-name
        replace_with_participant_external_ids=False,
        # pylint: disable=invalid-name
        replace_with_family_external_ids=False,
        empty_participant_value='',
        include_header=False,
    ) -> List[List[Optional[str]]]:
        """
        Generate pedigree file for ALL families in project
        (unless internal_family_ids is specified).

        Use internal IDs unless specific options are specified.
        """

        # this is important because a PED file MUST be ordered like this
        ordered_keys = [
            'family_id',
            'participant_id',
            'paternal_participant_id',
            'maternal_participant_id',
            'sex',
            'affected',
        ]
        pid_fields = {
            'participant_id',
            'paternal_participant_id',
            'maternal_participant_id',
        }

        rows = await self.fptable.get_rows(project=project, family_ids=family_ids)
        pmap, fmap = {}, {}
        if replace_with_participant_external_ids:
            participant_ids = set(
                s
                for r in rows
                for s in [r[pfield] for pfield in pid_fields]
                if s is not None
            )
            ptable = ParticipantTable(connection=self.connection)
            pmap = await ptable.get_id_map_by_internal_ids(list(participant_ids))

        if replace_with_family_external_ids:
            family_ids = list(
                set(r['family_id'] for r in rows if r['family_id'] is not None)
            )
            fmap = await self.ftable.get_id_map_by_internal_ids(list(family_ids))

        formatted_rows = []
        if include_header:
            formatted_rows.append(PedRow.row_header())

        for row in rows:
            formatted_row = []
            for field in ordered_keys:
                value = row[field]
                if field == 'family_id':
                    formatted_row.append(fmap.get(value, value))
                elif field in pid_fields:
                    formatted_row.append(
                        pmap.get(value, value) or empty_participant_value
                    )
                else:
                    formatted_row.append(value)
            formatted_rows.append(formatted_row)

        return formatted_rows

    async def get_participant_family_map(
        self, participant_ids: List[int], check_project_ids=False
    ):
        """Get participant family map"""

        fptable = FamilyParticipantTable(self.connection)
        projects, family_map = await fptable.get_participant_family_map(
            participant_ids=participant_ids
        )

        if check_project_ids:
            raise NotImplementedError(f'Must check specified projects: {projects}')

        return family_map

    async def import_pedigree(
        self,
        header: Optional[List[str]],
        rows: List[List[str]],
        create_missing_participants=False,
        perform_sex_check=True,
    ):
        """
        Import pedigree file
        """
        if header is None:
            _header = PedRow.default_header()
        else:
            _header = PedRow.parse_header_order(header)

        if len(rows) == 0:
            return None

        max_row_length = len(rows[0])
        if max_row_length > len(_header):
            raise ValueError(
                f"The parsed header {_header} isn't long enough "
                f'to cover row length ({len(_header)} < {len(rows[0])})'
            )
        if len(_header) > max_row_length:
            _header = _header[:max_row_length]

        pedrows: List[PedRow] = [
            PedRow(**{_header[i]: r[i] for i in range(len(_header))}) for r in rows
        ]
        # this validates a lot of the pedigree too
        pedrows = PedRow.order(pedrows)
        if perform_sex_check:
            PedRow.validate_sexes(pedrows, throws=True)

        external_family_ids = set(r.family_id for r in pedrows)
        # get set of all individual, paternal, maternal participant ids
        external_participant_ids = set(
            pid
            for r in pedrows
            for pid in [r.individual_id, r.paternal_id, r.maternal_id]
            if pid
        )

        participant_table = ParticipantLayer(self.connection)

        external_family_id_map = await self.ftable.get_id_map_by_external_ids(
            list(external_family_ids),
            project=self.connection.project,
            allow_missing=True,
        )
        missing_external_family_ids = [
            f for f in external_family_ids if f not in external_family_id_map
        ]
        external_participant_ids_map = await participant_table.get_id_map_by_external_ids(
            list(external_participant_ids),
            project=self.connection.project,
            # Allow missing participants if we're creating them
            allow_missing=create_missing_participants,
        )

        async with self.connection.connection.transaction():

            if create_missing_participants:
                missing_participant_ids = set(external_participant_ids) - set(
                    external_participant_ids_map
                )
                for row in pedrows:
                    if row.individual_id not in missing_participant_ids:
                        continue
                    external_participant_ids_map[
                        row.individual_id
                    ] = await participant_table.create_participant(
                        external_id=row.individual_id, reported_sex=row.sex
                    )

            for external_family_id in missing_external_family_ids:
                internal_family_id = await self.ftable.create_family(
                    external_id=external_family_id,
                    description=None,
                    coded_phenotype=None,
                )
                external_family_id_map[external_family_id] = internal_family_id

            # now let's map participants back

            insertable_rows = [
                {
                    'family_id': external_family_id_map[row.family_id],
                    'participant_id': external_participant_ids_map[row.individual_id],
                    'paternal_participant_id': external_participant_ids_map.get(
                        row.paternal_id
                    ),
                    'maternal_participant_id': external_participant_ids_map.get(
                        row.maternal_id
                    ),
                    'affected': row.affected,
                    'notes': row.notes,
                }
                for row in pedrows
            ]

            await participant_table.update_participants(
                participant_ids=[
                    external_participant_ids_map[row.individual_id] for row in pedrows
                ],
                reported_sexes=[row.sex for row in pedrows],
            )
            await self.fptable.create_rows(insertable_rows)

        return True

    async def import_families(
        self, headers: Optional[List[str]], rows: List[List[str]]
    ):
        """Import a family table"""
        ordered_headers = [
            'Family ID',
            'Display Name',
            'Description',
            'Coded Phenotype',
        ]
        _headers = headers or ordered_headers[: len(rows[0])]
        lheaders = [k.lower() for k in _headers]
        key_map = {
            'externalId': {'family_id', 'family id', 'familyid'},
            'displayName': {'display name', 'displayname', 'display_name'},
            'description': {'description'},
            'phenotype': {
                'coded phenotype',
                'phenotype',
                'codedphenotype',
                'coded_phenotype',
            },
        }

        def get_idx_for_header(header) -> Optional[int]:
            return next(
                iter(idx for idx, key in enumerate(lheaders) if key in key_map[header]),
                None,
            )

        external_identifier_idx = get_idx_for_header('externalId')
        display_name_idx = get_idx_for_header('displayName')
        description_idx = get_idx_for_header('description')
        phenotype_idx = get_idx_for_header('phenotype')

        # replace empty strings with None
        def replace_empty_string_with_none(val):
            """Don't set as empty string, prefer to set as null"""
            return None if val == '' else val

        rows = [[replace_empty_string_with_none(el) for el in r] for r in rows]

        empty = [None] * len(rows)

        def select_columns(col1: Optional[int], col2: Optional[int] = None):
            """
            - If col1 and col2 is None, return [None] * len(rows)
            - if either col1 or col2 is not None, return that column
            - else, return a mixture of column col1 | col2 if set
            """
            if col1 is None and col2 is None:
                # if col1 AND col2 is NONE
                return empty
            if col1 is not None and col2 is None:
                # if only col1 is set
                return [r[col1] for r in rows]
            if col2 is not None and col1 is None:
                # if only col2 is set
                return [r[col2] for r in rows]
            # if col1 AND col2 are not None
            assert col1 is not None and col2 is not None
            return [r[col1] if r[col1] is not None else r[col2] for r in rows]

        await self.ftable.insert_or_update_multiple_families(
            external_ids=select_columns(external_identifier_idx, display_name_idx),
            descriptions=select_columns(description_idx),
            coded_phenotypes=select_columns(phenotype_idx),
        )
        return True
