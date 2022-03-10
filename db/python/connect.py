# pylint: disable=unused-import
#
"""
Code for connecting to Postgres database
"""
import json
import logging
import os
from typing import Optional

# import asyncio
import databases

from db.python.tables.project import ProjectPermissionsTable

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def to_db_json(val):
    """Convert val to json for DB"""
    # return psycopg2.extras.Json(val)
    return json.dumps(val)


class Connection:
    """Stores a DB connection, project and author"""

    def __init__(
        self,
        connection: databases.Database,
        project: Optional[int],
        author: str,
    ):
        self.connection: databases.Database = connection
        self.project: Optional[int] = project
        self.author: str = author

    def assert_requires_project(self):
        """Assert the project is set, or return an exception"""
        if self.project is None:
            raise Exception(
                'An internal error has occurred when passing the project context, '
                'please send this stacktrace to your system administrator'
            )


class NotFoundError(Exception):
    """Custom error when you can't find something"""


class DatabaseConfiguration:
    """Class to hold information about a MySqlConfiguration"""

    def __init__(
        self,
        dbname,
        host=None,
        port=None,
        username=None,
        password=None,
    ):
        self.dbname = dbname
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    @staticmethod
    def dev_config() -> 'DatabaseConfiguration':
        """Dev config for local database with name 'sm_dev'"""
        # consider pulling from env variables
        return DatabaseConfiguration(
            dbname=os.environ.get('SM_DEV_DB_NAME', 'sm_dev'),
            username=os.environ.get('SM_DEV_DB_USER', 'root'),
            password=os.environ.get('SM_DEV_DB_PASSWORD', ''),
            host=os.environ.get('SM_DEV_DB_HOST', '127.0.0.1'),
            port=str(os.environ.get('SM_DEV_DB_PORT', '3306')),
        )


class SMConnections:
    """Contains useful functions for connecting to the database"""

    # _connected = False
    # _connections: Dict[str, databases.Database] = {}
    # _admin_db: databases.Database = None

    _credentials: Optional[DatabaseConfiguration] = None

    @staticmethod
    def _get_config():
        if SMConnections._credentials:
            return SMConnections._credentials

        config = DatabaseConfiguration.dev_config()
        creds_from_env = os.getenv('SM_DBCREDS')
        if creds_from_env is not None:
            config = DatabaseConfiguration(**json.loads(creds_from_env))
            logger.info(f'Using supplied SM DB CREDS: {config.host}:{config.port}')

        SMConnections._credentials = config

        return SMConnections._credentials

    @staticmethod
    def make_connection(config: DatabaseConfiguration):
        """Create connection from dbname"""
        # the connection string will prepare pooling automatically
        return databases.Database(
            SMConnections.prepare_connection_string(
                host=config.host,
                database=config.dbname,
                username=config.username,
                password=config.password,
                port=config.port,
            )
        )

    @staticmethod
    def prepare_connection_string(
        host,
        database,
        username,
        password=None,
        port=None,
        # min_pool_size=5,
        # max_pool_size=20,
    ):
        """Prepares the connection string for mysql / mariadb"""

        _host = host or 'localhost'
        u_p = username

        if password:
            u_p += f':{password}'
        if port:
            _host += f':{port}'

        options = {}  # {'min_size': min_pool_size, 'max_size': max_pool_size}
        _options = '&'.join(f'{k}={v}' for k, v in options.items())

        url = f'mysql://{u_p}@{_host}/{database}?{_options}'

        return url

    @staticmethod
    async def connect():
        """Create connections to database"""

        # this will now not connect, new connection will be made on every request
        return False

    @staticmethod
    async def disconnect():
        """Disconnect from database"""

        return False

    @staticmethod
    async def _get_made_connection():
        credentials = SMConnections._get_config()

        if credentials is None:
            raise Exception(
                'The server has been misconfigured, please '
                'contact your system administrator'
            )

        conn = SMConnections.make_connection(credentials)
        await conn.connect()
        return conn

    @staticmethod
    async def get_connection(*, author: str, project_name: str, readonly: bool):
        """Get a db connection from a project and user"""
        # maybe it makes sense to perform permission checks here too
        logger.debug(f'Authenticate connection to {project_name} with "{author}"')

        conn = await SMConnections._get_made_connection()
        pt = ProjectPermissionsTable(connection=conn)

        project_id = await pt.get_project_id_from_name_and_user(
            user=author, project_name=project_name, readonly=readonly
        )

        return Connection(connection=conn, author=author, project=project_id)

    @staticmethod
    async def get_connection_no_project(author: str):
        """Get a db connection from a project and user"""
        # maybe it makes sense to perform permission checks here too
        logger.debug(f'Authenticate no-project connection with "{author}"')

        conn = await SMConnections._get_made_connection()

        # we don't authenticate project-less connection, but rely on the
        # the endpoint to validate the resources

        return Connection(connection=conn, author=author, project=None)


class DbBase:
    """Base class for table subclasses"""

    @classmethod
    async def from_project(cls, project, author, readonly: bool):
        """Create the Db object from a project with user details"""
        return cls(
            connection=await SMConnections.get_connection(
                project_name=project, author=author, readonly=readonly
            ),
        )

    def __init__(self, connection: Connection):
        if connection is None:
            raise Exception(
                f'No connection was provided to the table "{self.__class__.__name__}"'
            )
        if not isinstance(connection, Connection):
            raise Exception(
                f'Expected connection type Connection, received {type(connection)}, '
                f'did you mean to call self._connection?'
            )

        self._connection = connection
        self.connection: databases.Database = connection.connection
        self.author = connection.author
        self.project = connection.project

        if self.author is None:
            raise Exception(f'Must provide author to {self.__class__.__name__}')

    # piped from the connection
