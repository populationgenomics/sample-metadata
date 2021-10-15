#!/usr/bin/env python3
# pylint: disable=logging-not-lazy,subprocess-popen-preexec-fn,consider-using-with
import logging
from typing import Optional

import os
import signal
import tempfile
import shutil
import time
import subprocess

import requests

DOCKER_IMAGE = os.getenv('SM_DOCKER')
SCHEMA_URL = os.getenv('SM_SCHEMAURL', 'http://localhost:8000/openapi.json')
MODULE_NAME = 'sample_metadata.generated_api'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_if_server_is_accessible() -> bool:
    """Check if request to 'SCHEMA_URL' returns OK"""
    try:
        return requests.get(SCHEMA_URL).ok
    except requests.ConnectionError:
        return False


def start_server() -> Optional[subprocess.Popen]:
    """Start the API server, and return a process when it's started"""

    command = ['python', '-m', 'api.server']

    if DOCKER_IMAGE is not None:
        command = [
            'docker',
            'run',
            '-eSM_HOST=0.0.0.0',
            '-eSM_IGNORE_GCP_CREDENTIALS_ERROR=1',
            '-eSM_SKIP_DATABASE_CONNECTION=1',
            '-p8000:8000',
            DOCKER_IMAGE,
            *command,
        ]

    logger.info('Starting API server with: ' + ' '.join(command))

    _process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # The os.setsid() is passed in the argument preexec_fn so
        # it's run after the fork() and before  exec() to run the shell.
        preexec_fn=os.setsid,
    )

    for c in iter(_process.stdout.readline, 'b'):
        line = None
        if c:
            line = c.decode('utf-8').rstrip()
            if line is not None:
                logger.info('API: ' + line)
                if 'running on http' in line.lower():
                    # server has been started
                    logger.info('Started process')
                    return _process

        rc = _process.poll()
        if rc is not None:
            # the process exited early
            logger.error(f'Server exited with rc={rc}')
            if DOCKER_IMAGE is not None:
                logger.warning(
                    "If you're receiving a 'port is already allocated' message, "
                    "run 'docker ps' and make sure the container isn't already running."
                )
            raise SystemExit(1)

    return None


def generate_api_and_copy():
    """Get JSON from server"""
    tmpdir = tempfile.mkdtemp()
    command = [
        'openapi-generator',
        'generate',
        '-i',
        SCHEMA_URL,
        '-g',
        'python',
        '-o',
        tmpdir,
        '--package-name',
        MODULE_NAME,
        '--skip-validate-spec',
    ]
    # 5 attempts
    n_attempts = 1
    succeeded = False
    for i in range(n_attempts, 0, -1):
        try:
            stdout = subprocess.check_output(command)
            logger.info('Generated API: ' + str(stdout.decode()))
            succeeded = True
            break

        except subprocess.CalledProcessError as e:
            logger.warning(f'openapi generation failed, trying {i-1} more times: {e}')
            time.sleep(2)

    if not succeeded:
        return

    copy_files_from(tmpdir)

    shutil.rmtree(tmpdir)


def copy_files_from(tmpdir):
    """
    Copy a selection of API files generated from openapi-generator:

        FROM:   $tmpdir/sample_metadata
        TO:     ./sample_metadata/api

    This clears the ./sample_metadata/api folder except for 'files_to_ignore'.
    """

    files_to_ignore = {'configuration.py', 'README.md', 'model_utils.py'}

    module_dir = MODULE_NAME.replace('.', '/')
    dir_to_copy_to = module_dir  # should be relative to this script
    dir_to_copy_from = os.path.join(tmpdir, module_dir)

    if not os.path.exists(dir_to_copy_to):
        raise FileNotFoundError(
            f"Directory to copy to doesn't exist ({dir_to_copy_to})"
        )
    if not os.path.exists(dir_to_copy_from):
        raise FileNotFoundError(
            f"Directory to copy from doesn't exist ({dir_to_copy_from})"
        )

    # remove everything from dir_to_copy_to except those in files_to_ignore
    logger.info('Removing files from dest directory ' + dir_to_copy_to)
    for file_to_remove in os.listdir(dir_to_copy_to):
        if file_to_remove in files_to_ignore:
            continue
        path_to_remove = os.path.join(dir_to_copy_to, file_to_remove)
        if os.path.isdir(path_to_remove):
            shutil.rmtree(path_to_remove)
        else:
            os.remove(path_to_remove)

    files_to_copy = os.listdir(dir_to_copy_from)
    logger.info(f'Copying {len(files_to_copy)} files / directories to {dir_to_copy_to}')
    for file_to_copy in files_to_copy:
        if file_to_copy in files_to_ignore:
            continue

        path_to_copy = os.path.join(dir_to_copy_from, file_to_copy)
        output_path = os.path.join(dir_to_copy_to, file_to_copy)
        if os.path.isdir(path_to_copy):
            shutil.copytree(path_to_copy, output_path)
        else:
            shutil.copy(path_to_copy, output_path)


def main():
    """
    Generates installable python API using:
        - Start API server (if applicable);
        - Call openapi-generator to generate python API to temp folder;
        - Empty the 'sample_metadata' folder (except for some files);
        - Copy relevant files to 'sample_metadata' in CWD;
        - Stop the server (if applicable)

    """
    if check_if_server_is_accessible():
        logger.info(f'Using already existing server {SCHEMA_URL}')
        process = None
    else:
        process = start_server()

    try:
        generate_api_and_copy()
    # pylint: disable=broad-except
    except BaseException as e:
        logger.error(str(e))
        if process:
            pid = process.pid
            logger.info(f'Stopping self-managed server by sending sigkill to {pid}')
            os.killpg(
                os.getpgid(process.pid), signal.SIGTERM
            )  # Send the signal to all the process groups

        raise e

    if process:
        pid = process.pid
        logger.info(f'Stopping self-managed server by sending sigkill to {pid}')
        os.killpg(
            os.getpgid(process.pid), signal.SIGTERM
        )  # Send the signal to all the process groups


if __name__ == '__main__':
    main()
