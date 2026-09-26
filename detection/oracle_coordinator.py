from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from detection.oracle_node import OracleNode

if TYPE_CHECKING:
    from detection.soroban_publisher import SorobanPublisher

logger = logging.getLogger("ledgerlens.oracle_coordinator")


@dataclass
class QuorumSignature:
    message_bytes: bytes            # canonical message that was signed
    signatures: list[tuple[str, str]]  # [(public_key_hex, signature_hex), ...]
    signers_count: int
    threshold: int
    is_valid_quorum: bool           # True if signers_count >= threshold


@dataclass
class NodeLiveness:
    """Tracks heartbeat liveness for a single oracle node."""
    name: str
    last_heartbeat: float
    is_active: bool = True


class OracleCoordinator:
    """
    Coordinates threshold signatures across multiple OracleNodes.

    Also monitors node liveness via periodic heartbeats and automatically
    reconfigures the quorum threshold (within safe bounds) as the set of
    active nodes changes.
    """

    # A node is considered dead if no heartbeat arrives within this window.
    HEARTBEAT_TIMEOUT_SECONDS: float = 30.0
    # Minimum number of active nodes required to keep quorum achievable.
    MIN_ACTIVE_NODES: int = 2
    # Alert when active nodes drop to (or below) this many.
    ALERT_ACTIVE_NODES: int = 3

    def __init__(self, nodes: list[OracleNode], threshold: int = 3):
        if threshold > len(nodes):
            raise ValueError(f"Threshold {threshold} > node count {len(nodes)}")
        self.nodes = nodes
        self.threshold = threshold
        self._base_threshold = threshold
        now = time.monotonic()
        self._liveness: dict[str, NodeLiveness] = {
            node.name: NodeLiveness(name=node.name, last_heartbeat=now)
            for node in nodes
        }

    # ------------------------------------------------------------------
    # Heartbeat liveness monitoring
    # ------------------------------------------------------------------
    def record_heartbeat(self, node_name: str, timestamp: float | None = None) -> None:
        """Record a periodic heartbeat from an oracle node."""
        liveness = self._liveness.get(node_name)
        if liveness is None:
            logger.warning("Heartbeat from unknown oracle node: %s", node_name)
            return
        liveness.last_heartbeat = timestamp if timestamp is not None else time.monotonic()
        if not liveness.is_active:
            logger.info("Oracle node %s rejoined the active set", node_name)
        liveness.is_active = True

    def _refresh_liveness(self, now: float | None = None) -> None:
        """Mark nodes that missed the heartbeat window as inactive."""
        now = now if now is not None else time.monotonic()
        for liveness in self._liveness.values():
            if liveness.is_active and (now - liveness.last_heartbeat) > self.HEARTBEAT_TIMEOUT_SECONDS:
                liveness.is_active = False
                logger.warning(
                    "Oracle node %s missed heartbeat window (%.1fs); excluding from quorum",
                    liveness.name,
                    now - liveness.last_heartbeat,
                )

    def active_nodes(self, now: float | None = None) -> list[OracleNode]:
        """Return nodes currently considered live (heartbeating)."""
        self._refresh_liveness(now)
        return [node for node in self.nodes if self._liveness[node.name].is_active]

    # ------------------------------------------------------------------
    # Automatic quorum reconfiguration
    # ------------------------------------------------------------------
    def reconfigure_quorum(self, now: float | None = None) -> int:
        """
        Recompute the quorum threshold based on the active node count.

        The threshold is kept within safe bounds: never below MIN_ACTIVE_NODES
        and never above the number of active nodes (so quorum stays achievable).
        """
        active_count = len(self.active_nodes(now))
        if active_count < self.MIN_ACTIVE_NODES:
            logger.error(
                "Active oracle nodes (%d) below minimum required for quorum (%d)",
                active_count,
                self.MIN_ACTIVE_NODES,
            )
            new_threshold = self.MIN_ACTIVE_NODES
        else:
            new_threshold = min(self._base_threshold, active_count)
            new_threshold = max(new_threshold, self.MIN_ACTIVE_NODES)

        if new_threshold != self.threshold:
            logger.info(
                "Reconfiguring quorum threshold %d -> %d (active nodes: %d)",
                self.threshold,
                new_threshold,
                active_count,
            )
            self.threshold = new_threshold

        if active_count <= self.ALERT_ACTIVE_NODES:
            logger.warning(
                "Active oracle node count (%d) approaching minimum quorum requirement (%d)",
                active_count,
                self.MIN_ACTIVE_NODES,
            )
        return self.threshold

    def collect_signatures(
        self,
        wallet: str,
        asset_pair: str,
        score: int,
        benford_flag: bool,
        ml_flag: bool,
        timestamp: int,
        confidence: int,
        model_version: int,
    ) -> QuorumSignature:
        """Collect signatures from active nodes; stop after threshold is reached."""
        self.reconfigure_quorum()
        message = OracleNode._canonical_message(
            wallet,
            asset_pair,
            score,
            benford_flag,
            ml_flag,
            timestamp,
            confidence,
            model_version,
        )
        signatures = []
        for node in self.active_nodes():
            try:
                sig = node.sign_score_submission(
                    wallet,
                    asset_pair,
                    score,
                    benford_flag,
                    ml_flag,
                    timestamp,
                    confidence,
                    model_version,
                )
                signatures.append((node.public_key_hex, sig.hex()))
                if len(signatures) >= self.threshold:
                    break      # Short-circuit: quorum reached
            except Exception as e:
                logger.warning("Oracle %s failed to sign: %s", node.name, e)
        return QuorumSignature(
            message_bytes=message,
            signatures=signatures,
            signers_count=len(signatures),
            threshold=self.threshold,
            is_valid_quorum=len(signatures) >= self.threshold,
        )

    def submit_with_quorum(
        self,
        wallet: str,
        asset_pair: str,
        score: int,
        benford_flag: bool,
        ml_flag: bool,
        timestamp: int,
        confidence: int,
        model_version: int,
        publisher: "SorobanPublisher",
    ) -> bool:
        """Collects quorum signatures and forwards to the publisher."""
        quorum = self.collect_signatures(
            wallet,
            asset_pair,
            score,
            benford_flag,
            ml_flag,
            timestamp,
            confidence,
            model_version,
        )
        if not quorum.is_valid_quorum:
            logger.error(
                "Quorum not reached: %d/%d signatures",
                quorum.signers_count,
                self.threshold,
            )
            return False
        # Call oracle_aggregator Soroban contract
        return publisher.submit_with_quorum(
            wallet,
            asset_pair,
            score,
            benford_flag,
            ml_flag,
            timestamp,
            confidence,
            model_version,
            quorum,
        )
