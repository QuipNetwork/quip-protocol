"""Fork-aware block synchronizer.

Each call to :py:meth:`BlockSynchronizer.sync_blocks` runs four phases
against one peer-advertised tip at a time:

1. **Tip discovery** — survey every reachable peer for its latest
   block, group peers by head hash, rank groups by
   ``(height desc, peer_count desc, head_hash asc)``.
2. **Manifest fetch** — pick the best group, pull a Bitcoin-style
   ``(index, hash)`` manifest from up to two peers in it, cross-check
   overlapping entries, and paginate forward until the pinned head is
   covered.
3. **Block download** — fetch every manifest block via
   ``GET_BLOCK_BY_HASH`` in shuffled order and verify each response's
   index, own hash, and ``previous_hash`` linkage against the manifest
   (with the first entry's parent checked against the local chain).
4. **Commit** — hand blocks to the node's block processing queue in
   ascending index order, awaiting each future so the final gate is
   the existing ``Node.check_block`` validation path.

If any phase fails against the pinned group — manifest disagreement,
all peers exhausted into the session backoff set, linkage error — the
group is dropped and the next candidate is tried. The backoff set is
local to one sync session; a peer demoted for group A stays demoted
for group B within the same session.

The synchronizer runs entirely inside a single asyncio event loop;
concurrency is bounded by ``max_in_flight``.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from shared.block import Block
from shared.node_client import NodeClient
from shared.sync_messages import MAX_MANIFEST_ENTRIES

# Timeout for a single peer's tip-survey request. Kept short so slow
# peers don't stall the whole ranking phase.
TIP_SURVEY_TIMEOUT_SECONDS = 5.0

# Timeout for awaiting a single block's acceptance through the node's
# block processing queue.
COMMIT_TIMEOUT_SECONDS = 30.0


@dataclass
class TipGroup:
    """Set of peers advertising the same canonical chain head."""

    height: int
    head_hash: bytes
    peers: List[str]

    def __repr__(self) -> str:  # pragma: no cover - debug-only formatting
        return (
            f"TipGroup(height={self.height}, peers={len(self.peers)}, "
            f"head={self.head_hash.hex()[:8]})"
        )


@dataclass
class SyncResult:
    """Outcome of one sync attempt."""

    success: bool
    target_height: Optional[int] = None
    target_hash: Optional[bytes] = None
    committed: int = 0
    requested: int = 0
    downloaded: int = 0
    failed_block: Optional[int] = None
    elapsed: float = 0.0
    reason: Optional[str] = None
    groups_tried: int = 0
    peers_backed_off: int = 0

    def summary(self) -> str:
        if self.success:
            return (
                f"Synced {self.committed} blocks to height "
                f"{self.target_height} in {self.elapsed:.1f}s"
            )
        return (
            f"Sync failed after {self.groups_tried} tip group(s): "
            f"{self.reason or 'unknown'}"
        )


class BlockSynchronizer:
    """Fork-aware block synchronizer that commits to one peer tip per run."""

    # Upper bound on pagination rounds during manifest fetch. Each round
    # returns up to ``MAX_MANIFEST_ENTRIES`` entries, so this caps a
    # single sync at roughly 2048 * MAX_MANIFEST_PAGES blocks — plenty
    # of headroom for foreseeable chain lengths.
    MAX_MANIFEST_PAGES = 512

    # Max per-block retries before aborting phase 3 for the current group.
    MAX_BLOCK_RETRIES = 3

    def __init__(
        self,
        node_client: NodeClient,
        receive_block_queue: asyncio.Queue,
        local_tip: Callable[[], Block],
        local_locator: Callable[[], List[bytes]],
        local_get_block_by_hash: Callable[[bytes], Optional[Block]],
        logger: Optional[logging.Logger] = None,
        max_in_flight: int = 32,
    ):
        """
        Args:
            node_client: QUIC client carrying the current peer set.
            receive_block_queue: ``asyncio.Queue`` the node's block
                processor consumes; items are
                ``(block, future, force_reorg, source)`` tuples.
            local_tip: Returns the local chain's latest ``Block``.
            local_locator: Returns a tip-first locator of local block
                hashes (see ``shared.node.Node.build_locator``).
            local_get_block_by_hash: Lookup function over the local
                canonical chain; used to verify the first manifest
                entry's parent hash is a block we actually hold.
            logger: Logger instance (defaults to module logger).
            max_in_flight: Upper bound on concurrent block-by-hash
                requests during phase 3.
        """
        self.node_client = node_client
        self.receive_block_queue = receive_block_queue
        self.local_tip = local_tip
        self.local_locator = local_locator
        self.local_get_block_by_hash = local_get_block_by_hash
        self.logger = logger or logging.getLogger(__name__)
        self.max_in_flight = max_in_flight

    async def sync_blocks(
        self,
        start_index: int = 1,
        end_index: int = 0,
    ) -> SyncResult:
        """Run one fork-aware sync attempt.

        ``start_index`` is ignored — the common ancestor is derived from
        the local locator. ``end_index``, if non-zero, is treated as an
        upper-bound hint: tips reporting a height strictly greater than
        this are skipped to avoid runaway downloads against a buggy or
        hostile peer.
        """
        t0 = time.monotonic()
        local_tip = self.local_tip()
        local_height = local_tip.header.index

        candidates, surveyed, failed = await self._discover_tips(
            local_height=local_height,
            max_height_hint=end_index if end_index > 0 else None,
        )
        if not candidates:
            # Distinguish "peers responded, none has a longer chain"
            # (genuinely in sync) from "every surveyed peer errored out"
            # (unknown state; do not mark ourselves synchronized).
            if surveyed > 0 and failed == surveyed:
                return SyncResult(
                    success=False,
                    elapsed=time.monotonic() - t0,
                    reason=f"tip survey failed for all {surveyed} peers",
                )
            return SyncResult(
                success=True,
                elapsed=time.monotonic() - t0,
                reason="no peer advertises a longer chain",
            )

        peer_backoff: Set[str] = set()
        groups_tried = 0

        for original in candidates:
            groups_tried += 1
            live_peers = [p for p in original.peers if p not in peer_backoff]
            if not live_peers:
                self.logger.info(
                    f"Skipping {original}: all peers already backed off"
                )
                continue
            group = TipGroup(original.height, original.head_hash, live_peers)
            self.logger.info(
                f"Pinning {group} (candidate {groups_tried}/{len(candidates)})"
            )

            manifest = await self._fetch_manifest(group, peer_backoff)
            if manifest is None or not manifest:
                self.logger.warning(
                    f"Manifest fetch failed for {group}; trying next group"
                )
                continue

            downloaded = await self._download_blocks(group, manifest, peer_backoff)
            if downloaded is None:
                self.logger.warning(
                    f"Block download failed for {group}; trying next group"
                )
                continue

            committed, failed_idx = await self._commit(manifest, downloaded)
            if failed_idx is not None:
                return SyncResult(
                    success=False,
                    target_height=group.height,
                    target_hash=group.head_hash,
                    requested=len(manifest),
                    downloaded=len(downloaded),
                    committed=committed,
                    failed_block=failed_idx,
                    elapsed=time.monotonic() - t0,
                    reason=f"commit rejected block {failed_idx}",
                    groups_tried=groups_tried,
                    peers_backed_off=len(peer_backoff),
                )

            return SyncResult(
                success=True,
                target_height=group.height,
                target_hash=group.head_hash,
                requested=len(manifest),
                downloaded=len(downloaded),
                committed=committed,
                elapsed=time.monotonic() - t0,
                groups_tried=groups_tried,
                peers_backed_off=len(peer_backoff),
            )

        return SyncResult(
            success=False,
            elapsed=time.monotonic() - t0,
            reason=f"all {groups_tried} tip groups exhausted",
            groups_tried=groups_tried,
            peers_backed_off=len(peer_backoff),
        )

    # ------------------------------------------------------------------
    # Phase 1 — tip discovery
    # ------------------------------------------------------------------

    async def _discover_tips(
        self,
        local_height: int,
        max_height_hint: Optional[int],
    ) -> Tuple[List[TipGroup], int, int]:
        """Query every peer for its tip and group peers by head hash.

        Returns ``(groups, surveyed, failed)``. ``failed`` counts peers
        whose tip query raised or returned ``None``; callers use it to
        distinguish "we surveyed peers and nobody has a longer chain"
        from "every peer errored so we don't know if we're behind".
        """
        peers = list(self.node_client.peers.keys())
        if not peers:
            return [], 0, 0

        tasks = [self._fetch_peer_tip(peer) for peer in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        groups: Dict[bytes, TipGroup] = {}
        failed = 0
        for peer, result in zip(peers, results):
            if isinstance(result, Exception) or result is None:
                failed += 1
                continue
            height, head_hash = result
            if height <= local_height:
                continue
            if max_height_hint is not None and height > max_height_hint:
                self.logger.info(
                    f"Skipping {peer} tip height {height}: exceeds hint "
                    f"{max_height_hint}"
                )
                continue
            g = groups.get(head_hash)
            if g is None:
                groups[head_hash] = TipGroup(height, head_hash, [peer])
            else:
                g.peers.append(peer)

        sorted_groups = sorted(
            groups.values(),
            key=lambda g: (-g.height, -len(g.peers), g.head_hash),
        )
        return sorted_groups, len(peers), failed

    async def _fetch_peer_tip(self, peer: str) -> Optional[Tuple[int, bytes]]:
        """Return ``(height, head_hash)`` for a peer, or ``None`` on failure.

        Timeouts are expected under partial network reachability and are
        logged at DEBUG. Any other exception is logged at WARNING so a
        programming/codec bug in the tip-survey path is discoverable
        rather than silently masquerading as a non-advancing peer.
        """
        try:
            block = await asyncio.wait_for(
                self.node_client.get_peer_block(peer, 0),
                timeout=TIP_SURVEY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.logger.debug(f"Tip survey for {peer} timed out")
            return None
        except Exception:
            self.logger.warning(
                f"Tip survey for {peer} failed", exc_info=True
            )
            return None
        if block is None or block.hash is None:
            return None
        return (block.header.index, block.hash)

    # ------------------------------------------------------------------
    # Phase 2 — manifest fetch
    # ------------------------------------------------------------------

    async def _fetch_manifest(
        self,
        group: TipGroup,
        peer_backoff: Set[str],
    ) -> Optional[List[Tuple[int, bytes]]]:
        """Fetch entries from common ancestor to pinned head, paginated.

        Cross-checks against a second peer when available. Demotes both
        peers on disagreement.
        """
        manifest: List[Tuple[int, bytes]] = []
        locator = self.local_locator()
        seen_indices: Set[int] = set()

        for _ in range(self.MAX_MANIFEST_PAGES):
            live = [p for p in group.peers if p not in peer_backoff]
            if not live:
                self.logger.warning(
                    f"No live peers remain in {group} during manifest fetch"
                )
                return None

            primary = random.choice(live)
            primary_entries = await self._fetch_manifest_page(
                primary, locator, peer_backoff
            )
            if primary_entries is None:
                continue  # primary demoted; retry with another peer
            if not primary_entries:
                break  # server has nothing else to give us

            secondaries = [p for p in live if p != primary]
            if secondaries:
                secondary = random.choice(secondaries)
                secondary_entries = await self._fetch_manifest_page(
                    secondary, locator, peer_backoff
                )
                if (
                    secondary_entries is not None
                    and secondary_entries
                    and not self._manifests_agree(primary_entries, secondary_entries)
                ):
                    self.logger.warning(
                        f"Manifest disagreement in {group}: demoting "
                        f"{primary} and {secondary}"
                    )
                    peer_backoff.add(primary)
                    peer_backoff.add(secondary)
                    return None

            # Append only entries we haven't already accepted (pagination
            # overlap is possible if peers return overlapping ranges).
            new_entries = [
                (idx, h) for (idx, h) in primary_entries if idx not in seen_indices
            ]
            if not new_entries:
                # No forward progress; avoid infinite loop.
                break
            for idx, h in new_entries:
                seen_indices.add(idx)
            manifest.extend(new_entries)

            last_idx, last_hash = manifest[-1]
            if last_idx >= group.height and last_hash == group.head_hash:
                break

            # The peer's chain may have grown since the tip survey.
            # If the manifest already covers the pinned height (even
            # with a different head because the peer mined further),
            # stop paginating — we have enough.
            if last_idx >= group.height:
                break

            # Advance locator: push the latest confirmed hash to the front
            # so the next page starts there.
            locator = [last_hash] + locator

        if not manifest:
            self.logger.warning(
                f"Manifest for {group} is empty"
            )
            return None

        # Check if the pinned head appears at the expected height.
        manifest_by_idx = {idx: h for idx, h in manifest}
        pinned_hash = manifest_by_idx.get(group.height)

        if pinned_hash == group.head_hash:
            # Exact match — truncate any entries beyond the pinned head
            # that arrived because the peer's chain grew during fetch.
            manifest = [(idx, h) for idx, h in manifest if idx <= group.height]
            return manifest

        if pinned_hash is not None:
            # The manifest covers the pinned height but with a different
            # hash — the peer's chain extended and potentially reorged.
            # Accept what we have up to the manifest's actual end;
            # update the group so downstream phases use the real tip.
            self.logger.info(
                f"Manifest for {group} extended to height "
                f"{manifest[-1][0]} (peer chain grew during fetch)"
            )
            group.height = manifest[-1][0]
            group.head_hash = manifest[-1][1]
            return manifest

        self.logger.warning(
            f"Manifest for {group} did not reach advertised height "
            f"{group.height}"
        )
        return None

    async def _fetch_manifest_page(
        self,
        peer: str,
        locator: List[bytes],
        peer_backoff: Set[str],
    ) -> Optional[List[Tuple[int, bytes]]]:
        """Fetch one manifest page. Demotes the peer on transport errors."""
        try:
            entries = await self.node_client.get_chain_manifest(
                peer, locator, MAX_MANIFEST_ENTRIES
            )
        except Exception as e:
            self.logger.warning(f"Manifest request to {peer} raised: {e}")
            peer_backoff.add(peer)
            return None
        if entries is None:
            peer_backoff.add(peer)
            return None
        return entries

    @staticmethod
    def _manifests_agree(
        a: List[Tuple[int, bytes]],
        b: List[Tuple[int, bytes]],
    ) -> bool:
        """Return True iff two manifest pages agree on overlapping indices."""
        a_by_idx = dict(a)
        for idx, h in b:
            if a_by_idx.get(idx, h) != h:
                return False
        return True

    # ------------------------------------------------------------------
    # Phase 3 — block download with linkage audit
    # ------------------------------------------------------------------

    async def _download_blocks(
        self,
        group: TipGroup,
        manifest: List[Tuple[int, bytes]],
        peer_backoff: Set[str],
    ) -> Optional[Dict[int, Block]]:
        """Fetch every manifest block via ``GET_BLOCK_BY_HASH`` in shuffled order.

        Returns ``None`` if the group can't fully serve the manifest —
        all peers demoted or a single block exhausts its retry budget.
        """
        manifest_by_idx: Dict[int, bytes] = dict(manifest)
        first_idx = manifest[0][0]
        work: List[Tuple[int, bytes]] = list(manifest)
        random.shuffle(work)
        downloaded: Dict[int, Block] = {}
        retries: Dict[int, int] = {}
        failure = {"aborted": False}

        async def worker() -> None:
            while not failure["aborted"]:
                if not work:
                    return
                idx, block_hash = work.pop()
                block = await self._try_fetch_block(
                    group=group,
                    idx=idx,
                    block_hash=block_hash,
                    manifest_by_idx=manifest_by_idx,
                    first_idx=first_idx,
                    peer_backoff=peer_backoff,
                )
                if block is not None:
                    downloaded[idx] = block
                    continue
                retries[idx] = retries.get(idx, 0) + 1
                if retries[idx] >= self.MAX_BLOCK_RETRIES:
                    self.logger.error(
                        f"Block {idx} exhausted {self.MAX_BLOCK_RETRIES} "
                        "retries; aborting group"
                    )
                    failure["aborted"] = True
                    return
                if not [p for p in group.peers if p not in peer_backoff]:
                    failure["aborted"] = True
                    return
                work.append((idx, block_hash))
                await asyncio.sleep(0.05)

        workers = [asyncio.create_task(worker()) for _ in range(self.max_in_flight)]
        await asyncio.gather(*workers)

        if failure["aborted"] or len(downloaded) < len(manifest):
            return None
        return downloaded

    async def _try_fetch_block(
        self,
        group: TipGroup,
        idx: int,
        block_hash: bytes,
        manifest_by_idx: Dict[int, bytes],
        first_idx: int,
        peer_backoff: Set[str],
    ) -> Optional[Block]:
        """Single attempt to fetch block ``(idx, block_hash)`` with full linkage check."""
        live = [p for p in group.peers if p not in peer_backoff]
        if not live:
            return None
        peer = random.choice(live)

        try:
            block = await self.node_client.get_peer_block_by_hash(peer, block_hash)
        except Exception as e:
            self.logger.warning(
                f"Block-by-hash from {peer} (idx {idx}) raised: {e}; demoting"
            )
            peer_backoff.add(peer)
            return None

        if block is None:
            self.logger.warning(
                f"Peer {peer} returned NOT_FOUND for idx {idx}; demoting"
            )
            peer_backoff.add(peer)
            return None

        if block.header.index != idx:
            self.logger.warning(
                f"Peer {peer} returned index {block.header.index} "
                f"for expected {idx}; demoting"
            )
            peer_backoff.add(peer)
            return None

        if block.hash != block_hash:
            self.logger.warning(
                f"Peer {peer} returned wrong-hash block at idx {idx}; demoting"
            )
            peer_backoff.add(peer)
            return None

        if idx == first_idx:
            parent = self.local_get_block_by_hash(block.header.previous_hash)
            if parent is None:
                self.logger.warning(
                    f"Peer {peer}: first manifest block's parent "
                    f"{block.header.previous_hash.hex()[:8]} is not on our "
                    "local chain; demoting"
                )
                peer_backoff.add(peer)
                return None
        else:
            expected_parent_hash = manifest_by_idx[idx - 1]
            if block.header.previous_hash != expected_parent_hash:
                self.logger.warning(
                    f"Peer {peer}: block {idx} previous_hash mismatches "
                    "manifest; demoting"
                )
                peer_backoff.add(peer)
                return None

        return block

    # ------------------------------------------------------------------
    # Phase 4 — commit
    # ------------------------------------------------------------------

    async def _commit(
        self,
        manifest: List[Tuple[int, bytes]],
        downloaded: Dict[int, Block],
    ) -> Tuple[int, Optional[int]]:
        """Push downloaded blocks through the node's processing queue.

        Returns ``(committed_count, first_failed_index)``. On success,
        ``first_failed_index`` is ``None``.
        """
        committed = 0
        for idx, _ in manifest:
            block = downloaded[idx]
            future: asyncio.Future = asyncio.Future()
            self.receive_block_queue.put_nowait(
                (block, future, True, "sync")
            )
            try:
                accepted = await asyncio.wait_for(
                    future, timeout=COMMIT_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Block {idx} commit timed out")
                return committed, idx
            if not accepted:
                self.logger.error(f"Block {idx} rejected during commit")
                return committed, idx
            committed += 1
        return committed, None
