"""In-memory user store backing the auth and user endpoints.

The API originally shipped with mock authentication: registration discarded
the password hash, any password could log in, and the profile endpoints
returned a hardcoded example user. This module gives those endpoints real
behavior for the lifetime of the process, using the real password hashing
and verification helpers from backend.app.core.security.

Persistence across restarts belongs in the database layer
(backend/app/core/database.py). The functions here are the interface the
routers use, so a database-backed implementation can replace the dicts
without changing any endpoint code.
"""
import itertools
from typing import Dict, Optional

from backend.app.core.security import get_password_hash, verify_password


class DuplicateUserError(Exception):
    """Raised when a username or email is already registered."""


_users_by_username: Dict[str, dict] = {}
_id_counter = itertools.count(1)


def _public(user: dict) -> dict:
    """Return the user record without the password hash."""
    return {key: value for key, value in user.items() if key != "hashed_password"}


def create_user(email: str, username: str, password: str,
                full_name: Optional[str] = None) -> dict:
    """Create and store a new user. Raises DuplicateUserError on conflicts."""
    if username in _users_by_username:
        raise DuplicateUserError(f"Username '{username}' is already registered")
    if any(u["email"] == email for u in _users_by_username.values()):
        raise DuplicateUserError(f"Email '{email}' is already registered")

    user = {
        "id": next(_id_counter),
        "email": email,
        "username": username,
        "full_name": full_name,
        "is_active": True,
        "hashed_password": get_password_hash(password),
    }
    _users_by_username[username] = user
    return _public(user)


def get_user(username: str) -> Optional[dict]:
    """Return the public record for a username, or None."""
    user = _users_by_username.get(username)
    return _public(user) if user else None


def authenticate(username: str, password: str) -> Optional[dict]:
    """Return the public user record if the password verifies, else None."""
    user = _users_by_username.get(username)
    if user is None or not user["is_active"]:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return _public(user)


def update_user(username: str, full_name: Optional[str] = None,
                email: Optional[str] = None) -> Optional[dict]:
    """Update a user's profile fields. Returns the updated record or None.

    Raises DuplicateUserError if the new email belongs to another user.
    """
    user = _users_by_username.get(username)
    if user is None:
        return None
    if email is not None and email != user["email"]:
        if any(u["email"] == email for u in _users_by_username.values()):
            raise DuplicateUserError(f"Email '{email}' is already registered")
        user["email"] = email
    if full_name is not None:
        user["full_name"] = full_name
    return _public(user)
