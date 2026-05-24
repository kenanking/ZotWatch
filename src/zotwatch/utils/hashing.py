"""Hashing utilities for ZotWatch."""

import hashlib


def hash_content(*parts: str) -> str:
    """Generate SHA256 hash from content parts.

    A null byte separator is inserted between parts to prevent collisions
    like hash_content("ab", "cd") == hash_content("a", "bcd").
    """
    sha = hashlib.sha256()
    for i, part in enumerate(parts):
        if part:
            if i > 0:
                sha.update(b"\x00")
            sha.update(part.encode("utf-8"))
    return sha.hexdigest()


__all__ = ["hash_content"]
