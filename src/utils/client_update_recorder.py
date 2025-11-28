import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class ClientUpdateRecorder:
    """Utility that persists raw client updates for offline analysis."""

    def __init__(
        self,
        base_dir: str,
        config: Dict[str, Any] | None = None,
        experiment_context: Dict[str, Any] | None = None,
        malicious_client_ids: Sequence[int] | None = None,
    ) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.base_path: Path | None = None
        self.metadata_path: Path | None = None
        self.metadata: Dict[str, Any] = {}

        if not self.enabled:
            return

        self.compression = config.get("compression", "npz")
        self.keep_float32 = bool(config.get("force_float32", True))
        subdir = config.get("output_subdir", "client_updates")
        self.base_path = Path(base_dir) / subdir
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.base_path / "metadata.json"
        default_context = dict(experiment_context or {})
        default_context["malicious_client_ids"] = list(malicious_client_ids or [])
        self.metadata = default_context
        self._persist_metadata()

    def _persist_metadata(self) -> None:
        if not self.enabled or not self.metadata_path:
            return
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def update_malicious_client_ids(self, client_ids: Sequence[int] | None) -> None:
        if not self.enabled:
            return
        self.metadata["malicious_client_ids"] = list(client_ids or [])
        self._persist_metadata()

    def _maybe_cast(self, array: np.ndarray) -> np.ndarray:
        if not self.keep_float32:
            return array
        if array.dtype == np.float32:
            return array
        if np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float32)
        return array

    def record_round(
        self,
        round_index: int,
        updates: List[Tuple[List[np.ndarray], int]],
        client_ids: List[int],
        extra_metadata: Dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self.base_path:
            return
        if not updates:
            return

        round_dir = self.base_path / f"round_{round_index:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        round_manifest: Dict[str, Any] = {
            "round": round_index,
            "num_clients": len(client_ids),
            "client_ids": client_ids,
        }
        if extra_metadata:
            round_manifest.update(extra_metadata)

        manifest_path = round_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(round_manifest, mf, indent=2)

        for client_id, (weights, num_examples) in zip(client_ids, updates):
            client_path = round_dir / f"client_{client_id:04d}.{self.compression}"
            arrays = {f"layer_{idx}": self._maybe_cast(layer) for idx, layer in enumerate(weights)}
            arrays["num_examples"] = np.array([num_examples], dtype=np.int64)
            if self.compression == "npz":
                np.savez_compressed(client_path, **arrays)
            else:
                np.savez(client_path, **arrays)