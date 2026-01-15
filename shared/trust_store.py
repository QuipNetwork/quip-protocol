"""
TOFU (Trust On First Use) certificate fingerprint store.

Stores peer certificate fingerprints in SQLite for secure node-to-node communication.
On first connection, the peer's certificate fingerprint is stored. On subsequent
connections, the fingerprint is verified against the stored value.
"""

import asyncio
import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class TofuResult(Enum):
    """Result of TOFU fingerprint verification."""
    NEW = "new"           # First time seeing this peer, fingerprint stored
    MATCH = "match"       # Fingerprint matches stored value
    MISMATCH = "mismatch" # Fingerprint does NOT match - possible MITM attack


@dataclass
class TrustedPeer:
    """Information about a trusted peer."""
    peer_address: str
    fingerprint: str
    first_seen: int
    last_seen: int
    connection_count: int


class TrustStore:
    """
    SQLite-based TOFU trust store for peer certificate fingerprints.

    Thread-safe implementation using connection-per-operation pattern
    since sqlite3 connections aren't thread-safe across async contexts.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS peer_fingerprints (
        peer_address TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        first_seen INTEGER NOT NULL,
        last_seen INTEGER NOT NULL,
        connection_count INTEGER DEFAULT 1
    );

    CREATE INDEX IF NOT EXISTS idx_last_seen ON peer_fingerprints(last_seen);
    """

    def __init__(self, db_path: str = "~/.quip/trust.db", logger: Optional[logging.Logger] = None):
        """
        Initialize the trust store.

        Args:
            db_path: Path to SQLite database file (supports ~ expansion)
            logger: Optional logger instance
        """
        self.db_path = os.path.expanduser(db_path)
        self.logger = logger or logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    async def initialize(self) -> None:
        """
        Initialize the database schema.

        Creates the database directory and tables if they don't exist.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Create directory if needed
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            # Create schema
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_schema)

            self._initialized = True
            self.logger.info(f"TOFU trust store initialized at {self.db_path}")

    def _init_schema(self) -> None:
        """Create database schema (runs in executor)."""
        conn = self._get_connection()
        try:
            conn.executescript(self.SCHEMA)
            conn.commit()
        finally:
            conn.close()

    async def get_fingerprint(self, peer_address: str) -> Optional[str]:
        """
        Get stored fingerprint for a peer.

        Args:
            peer_address: The peer's address (host:port)

        Returns:
            The stored fingerprint, or None if peer is not known
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_fingerprint_sync, peer_address)

    def _get_fingerprint_sync(self, peer_address: str) -> Optional[str]:
        """Synchronous fingerprint retrieval."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT fingerprint FROM peer_fingerprints WHERE peer_address = ?",
                (peer_address,)
            )
            row = cursor.fetchone()
            return row["fingerprint"] if row else None
        finally:
            conn.close()

    async def store_fingerprint(self, peer_address: str, fingerprint: str) -> bool:
        """
        Store fingerprint on first connection (TOFU).

        Args:
            peer_address: The peer's address (host:port)
            fingerprint: SHA-256 fingerprint of the peer's certificate

        Returns:
            True if stored successfully, False if peer already exists
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._store_fingerprint_sync, peer_address, fingerprint
        )

    def _store_fingerprint_sync(self, peer_address: str, fingerprint: str) -> bool:
        """Synchronous fingerprint storage."""
        now = int(time.time())
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT INTO peer_fingerprints
                   (peer_address, fingerprint, first_seen, last_seen, connection_count)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(peer_address) DO NOTHING""",
                (peer_address, fingerprint, now, now)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    async def verify_fingerprint(self, peer_address: str, fingerprint: str) -> TofuResult:
        """
        Verify peer fingerprint using TOFU model.

        On first connection, stores the fingerprint and returns NEW.
        On subsequent connections, compares against stored value.

        Args:
            peer_address: The peer's address (host:port)
            fingerprint: SHA-256 fingerprint of the peer's certificate

        Returns:
            TofuResult.NEW if first time seeing this peer (fingerprint stored)
            TofuResult.MATCH if fingerprint matches stored value
            TofuResult.MISMATCH if fingerprint does NOT match (possible MITM)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._verify_fingerprint_sync, peer_address, fingerprint
        )

    def _verify_fingerprint_sync(self, peer_address: str, fingerprint: str) -> TofuResult:
        """Synchronous fingerprint verification."""
        now = int(time.time())
        conn = self._get_connection()
        try:
            # Check if we have a stored fingerprint
            cursor = conn.execute(
                "SELECT fingerprint FROM peer_fingerprints WHERE peer_address = ?",
                (peer_address,)
            )
            row = cursor.fetchone()

            if row is None:
                # First time seeing this peer - store fingerprint (TOFU)
                conn.execute(
                    """INSERT INTO peer_fingerprints
                       (peer_address, fingerprint, first_seen, last_seen, connection_count)
                       VALUES (?, ?, ?, ?, 1)""",
                    (peer_address, fingerprint, now, now)
                )
                conn.commit()
                return TofuResult.NEW

            stored_fingerprint = row["fingerprint"]

            if stored_fingerprint == fingerprint:
                # Fingerprint matches - update last_seen and connection_count
                conn.execute(
                    """UPDATE peer_fingerprints
                       SET last_seen = ?, connection_count = connection_count + 1
                       WHERE peer_address = ?""",
                    (now, peer_address)
                )
                conn.commit()
                return TofuResult.MATCH
            else:
                # MISMATCH - possible MITM attack!
                return TofuResult.MISMATCH
        finally:
            conn.close()

    async def remove_fingerprint(self, peer_address: str) -> bool:
        """
        Remove a fingerprint (for manual trust reset).

        Use this when you want to re-establish trust with a peer
        (e.g., if they legitimately changed their certificate).

        Args:
            peer_address: The peer's address (host:port)

        Returns:
            True if a fingerprint was removed, False if peer wasn't found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._remove_fingerprint_sync, peer_address
        )

    def _remove_fingerprint_sync(self, peer_address: str) -> bool:
        """Synchronous fingerprint removal."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM peer_fingerprints WHERE peer_address = ?",
                (peer_address,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    async def list_trusted_peers(self) -> List[TrustedPeer]:
        """
        List all stored peer fingerprints.

        Returns:
            List of TrustedPeer objects ordered by last_seen (most recent first)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_trusted_peers_sync)

    def _list_trusted_peers_sync(self) -> List[TrustedPeer]:
        """Synchronous peer listing."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT peer_address, fingerprint, first_seen, last_seen, connection_count
                   FROM peer_fingerprints
                   ORDER BY last_seen DESC"""
            )
            return [
                TrustedPeer(
                    peer_address=row["peer_address"],
                    fingerprint=row["fingerprint"],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                    connection_count=row["connection_count"]
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    async def get_peer_info(self, peer_address: str) -> Optional[TrustedPeer]:
        """
        Get detailed information about a trusted peer.

        Args:
            peer_address: The peer's address (host:port)

        Returns:
            TrustedPeer object, or None if peer is not known
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_peer_info_sync, peer_address)

    def _get_peer_info_sync(self, peer_address: str) -> Optional[TrustedPeer]:
        """Synchronous peer info retrieval."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """SELECT peer_address, fingerprint, first_seen, last_seen, connection_count
                   FROM peer_fingerprints
                   WHERE peer_address = ?""",
                (peer_address,)
            )
            row = cursor.fetchone()
            if row:
                return TrustedPeer(
                    peer_address=row["peer_address"],
                    fingerprint=row["fingerprint"],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                    connection_count=row["connection_count"]
                )
            return None
        finally:
            conn.close()

    async def clear_all(self) -> int:
        """
        Clear all stored fingerprints.

        WARNING: This removes all trust relationships. Use with caution.

        Returns:
            Number of fingerprints removed
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._clear_all_sync)

    def _clear_all_sync(self) -> int:
        """Synchronous clear all."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM peer_fingerprints")
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()


def compute_certificate_fingerprint(cert_der: bytes) -> str:
    """
    Compute SHA-256 fingerprint of a DER-encoded certificate.

    Args:
        cert_der: DER-encoded certificate bytes

    Returns:
        Hex-encoded SHA-256 fingerprint (64 characters)
    """
    return hashlib.sha256(cert_der).hexdigest()
