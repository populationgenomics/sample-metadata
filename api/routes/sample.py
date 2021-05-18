# pylint: disable=unused-variable,unused-argument

from http import HTTPStatus

from flask import g
from flasgger import swag_from

from api.utils.request import JsonBlueprint
from api.schema.sample import SampleSchema
from db.python.tables.sample import SampleTable


def get_sample_blueprint(prefix):
    """Build blueprint / routes for sample API"""
    sample_api = JsonBlueprint('sample_api', __name__)
    project_prefix = prefix + '<project>/sample'

    @sample_api.route(project_prefix + '/<id_>', methods=['GET'])
    @swag_from(
        {
            'responses': {
                HTTPStatus.OK.value: {
                    'description': 'Welcome to the Flask Starter Kit',
                    'schema': SampleSchema,
                }
            },
            'parameters': [
                {'name': 'project', 'type': 'string', 'required': True, 'in': 'path'},
                {'name': 'id_', 'type': 'string', 'required': True, 'in': 'path'},
            ],
        }
    )
    def get_by_external_id(project, id_):
        """
        Get a sample by its external ID
        ---
        """

        st = SampleTable(g.connection, g.author)
        result = st.get_single_by_external_id(id_)
        return SampleSchema().dump(result), 200

    return sample_api
