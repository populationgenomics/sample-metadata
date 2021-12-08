import logging
from typing import Optional

from fastapi import Depends, Cookie, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.auth.transport import requests
from google.oauth2 import id_token

from db.python.connect import SMConnections, Connection
from api.utils.gcp import email_from_id_token

EXPECTED_AUDIENCE = '/projects/774248915715/global/backendServices/4615543739767186396'


def authenticate(
    token: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    x_goog_iap_jwt_assertion: Optional[str] = Cookie(None),
) -> str:
    """If a token (OR Google IAP auth jwt) is provided, return the email, else raise an Exception"""
    if token:
        return email_from_id_token(token.credentials)
    if x_goog_iap_jwt_assertion:
        return validate_iap_jwt_and_get_email(x_goog_iap_jwt_assertion)
    raise HTTPException(status_code=401, detail='Not authenticated')


async def dependable_get_write_project_connection(
    project: str, author: str = Depends(authenticate)
) -> Connection:
    """FastAPI handler for getting connection WITH project"""
    return await SMConnections.get_connection(
        project_name=project, author=author, readonly=False
    )


async def dependable_get_readonly_project_connection(
    project: str, author: str = Depends(authenticate)
) -> Connection:
    """FastAPI handler for getting connection WITH project"""
    return await SMConnections.get_connection(
        project_name=project, author=author, readonly=True
    )


async def dependable_get_connection(author: str = Depends(authenticate)):
    """FastAPI handler for getting connection withOUT project"""
    return await SMConnections.get_connection_no_project(author)


def validate_iap_jwt_and_get_email(iap_jwt):
    """Validate an IAP JWT. AND return email

    Args:
      iap_jwt: The contents of the X-Goog-IAP-JWT-Assertion header.
      expected_audience: The Signed Header JWT audience. See
          https://cloud.google.com/iap/docs/signed-headers-howto
          for details on how to get this value.

    Returns:
      (user_id, user_email, error_str).
    """

    try:
        decoded_jwt = id_token.verify_token(
            iap_jwt,
            requests.Request(),
            audience=EXPECTED_AUDIENCE,
            certs_url='https://www.gstatic.com/iap/verify/public_key',
        )
        return decoded_jwt['email']
    except Exception as e:
        logging.error(f'JWT validation error {e}')
        raise e


get_author = Depends(authenticate)
get_project_readonly_connection = Depends(dependable_get_readonly_project_connection)
get_project_write_connection = Depends(dependable_get_write_project_connection)
get_projectless_db_connection = Depends(dependable_get_connection)
