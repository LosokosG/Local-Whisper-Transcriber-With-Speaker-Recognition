"""Atomic, resumable run-state manifest with credential-safe diagnostics."""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import PipelineConfig
from .models import (
    BlockerRecord,
    ErrorSummary,
    StageStatus,
    from_utc_iso,
    utc_now_iso,
)
from .redaction import redact_value

_STAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_REQUIRED_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "updated_at",
        "input",
        "effective_config",
        "inferred_event",
        "tool_versions",
        "model_versions",
        "stages",
        "artifacts",
        "blockers",
        "manual_decisions",
    }
)
_REQUIRED_STAGE_FIELDS = frozenset(
    {
        "status",
        "attempts",
        "started_at",
        "finished_at",
        "artifacts",
        "artifact_hashes",
        "validation",
        "blocker_ids",
    }
)


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


class ManifestError(RuntimeError):
    """Base class for manifest-specific failures."""


class ManifestTransitionError(ManifestError):
    """Raised when a stage transition would violate the state machine."""


class ManifestPathError(ManifestError, ValueError):
    """Raised when a managed path could escape the run directory."""


class Manifest:
    """Load or create a `.work/manifest.json` run-state document."""

    schema_version = 1

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_dir: str | os.PathLike[str] | None = None,
        input_path: str | os.PathLike[str] | None = None,
        input_hash: str | None = None,
        input_sha256: str | None = None,
        media_metadata: Mapping[str, Any] | None = None,
        effective_config: PipelineConfig | Mapping[str, Any] | None = None,
        inferred_event: Mapping[str, Any] | None = None,
        tool_versions: Mapping[str, Any] | None = None,
        model_versions: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).resolve(strict=False)
        inferred_run_dir = (
            self.path.parent.parent
            if self.path.parent.name == ".work"
            else self.path.parent
        )
        self.run_dir = Path(run_dir or inferred_run_dir).resolve(strict=False)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        try:
            self.path.relative_to(self.run_dir)
        except ValueError as exc:
            raise ManifestPathError("manifest path must be inside the run directory") from exc

        if input_hash is not None and input_sha256 is not None:
            if input_hash != input_sha256:
                raise ValueError("input_hash and input_sha256 disagree")
        digest = input_hash if input_hash is not None else input_sha256
        config_data: Mapping[str, Any] | None
        if isinstance(effective_config, PipelineConfig):
            config_data = effective_config.to_dict()
        else:
            config_data = effective_config

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock():
            if self.path.exists():
                self._data = self._load()
                self._validate_existing_binding(input_path, digest, config_data)
            else:
                timestamp = utc_now_iso()
                normalized_input_path = (
                    self._normalize_input_path(input_path)
                    if input_path is not None
                    else None
                )
                self._data = {
                    "schema_version": self.schema_version,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "input": {
                        "path": normalized_input_path,
                        "sha256": digest,
                        "media": dict(media_metadata or {}),
                    },
                    "effective_config": dict(config_data or {}),
                    "inferred_event": dict(inferred_event or {}),
                    "tool_versions": dict(tool_versions or {}),
                    "model_versions": dict(model_versions or {}),
                    "stages": {},
                    "artifacts": {},
                    "blockers": [],
                    "manual_decisions": [],
                }
                self._commit(self._data)

    @classmethod
    def create(
        cls, path: str | os.PathLike[str], **kwargs: Any
    ) -> Manifest:
        return cls(path, **kwargs)

    @property
    def data(self) -> dict[str, Any]:
        with self._exclusive_lock():
            self._data = self._load()
            return copy.deepcopy(self._data)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _normalize_input_path(self, value: str | os.PathLike[str]) -> str:
        raw = os.fspath(value)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ManifestPathError("input path must be a non-empty text path")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.run_dir / path
        return str(path.resolve(strict=False))

    def _validate_existing_binding(
        self,
        input_path: str | os.PathLike[str] | None,
        input_hash: str | None,
        effective_config: Mapping[str, Any] | None,
    ) -> None:
        saved_input = self._data["input"]
        if input_path is not None:
            supplied_path = self._normalize_input_path(input_path)
            saved_path = saved_input.get("path")
            if (
                saved_path is None
                or self._normalize_input_path(saved_path) != supplied_path
            ):
                raise ManifestError("existing manifest input path does not match")
        if input_hash is not None and saved_input.get("sha256") != input_hash:
            raise ManifestError("existing manifest input hash does not match")
        if effective_config is not None:
            supplied_config = redact_value(copy.deepcopy(dict(effective_config)))
            if self._data.get("effective_config") != supplied_config:
                raise ManifestError("existing manifest effective config does not match")

    def _load(self) -> dict[str, Any]:
        try:
            document = json.loads(
                self.path.read_text(encoding="utf-8"),
                parse_constant=_reject_non_finite_json_constant,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ManifestError(f"cannot load manifest {self.path.name}: {exc}") from exc
        if not isinstance(document, dict):
            raise ManifestError("manifest root must be a JSON object")
        self._validate_document(document)
        return document

    @staticmethod
    def _missing_fields(
        value: Mapping[str, Any], required: frozenset[str], context: str
    ) -> None:
        missing = sorted(required - value.keys())
        if missing:
            raise ManifestError(
                f"{context} missing required fields: {', '.join(missing)}"
            )

    @staticmethod
    def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ManifestError(f"manifest {field} must be a JSON object")
        return value

    @staticmethod
    def _require_string(value: Any, field: str, *, nullable: bool = False) -> None:
        if nullable and value is None:
            return
        if not isinstance(value, str) or not value:
            expected = "a string or null" if nullable else "a non-empty string"
            raise ManifestError(f"manifest {field} must be {expected}")

    @staticmethod
    def _require_utc_timestamp(
        value: Any, field: str, *, nullable: bool = False
    ) -> None:
        if nullable and value is None:
            return
        if not isinstance(value, str) or not value:
            expected = "a UTC timestamp or null" if nullable else "a UTC timestamp"
            raise ManifestError(f"manifest {field} must be {expected}")
        if not (value.upper().endswith("Z") or value.endswith("+00:00")):
            raise ManifestError(f"manifest {field} must be serialized in UTC")
        try:
            from_utc_iso(value)
        except ValueError as exc:
            raise ManifestError(f"manifest {field} is not a valid UTC timestamp") from exc

    def _validate_document(self, document: Mapping[str, Any]) -> None:
        self._missing_fields(document, _REQUIRED_ROOT_FIELDS, "manifest")
        version = document["schema_version"]
        if type(version) is not int or version != self.schema_version:
            raise ManifestError(
                f"manifest schema_version must be exactly {self.schema_version}"
            )
        self._require_utc_timestamp(document["created_at"], "created_at")
        self._require_utc_timestamp(document["updated_at"], "updated_at")
        if from_utc_iso(document["updated_at"]) < from_utc_iso(
            document["created_at"]
        ):
            raise ManifestError("manifest updated_at cannot precede created_at")

        input_record = self._require_mapping(document["input"], "input")
        self._missing_fields(
            input_record, frozenset({"path", "sha256", "media"}), "manifest input"
        )
        self._require_string(input_record["path"], "input.path", nullable=True)
        if input_record["path"] is not None:
            try:
                self._normalize_input_path(input_record["path"])
            except (TypeError, ValueError) as exc:
                raise ManifestError(f"manifest input.path is unsafe: {exc}") from exc
        self._require_string(input_record["sha256"], "input.sha256", nullable=True)
        self._require_mapping(input_record["media"], "input.media")

        for field in (
            "effective_config",
            "inferred_event",
            "tool_versions",
            "model_versions",
        ):
            self._require_mapping(document[field], field)
        stages = self._require_mapping(document["stages"], "stages")
        artifacts = self._require_mapping(document["artifacts"], "artifacts")
        blockers = document["blockers"]
        decisions = document["manual_decisions"]
        if not isinstance(blockers, list):
            raise ManifestError("manifest blockers must be a JSON array")
        if not isinstance(decisions, list):
            raise ManifestError("manifest manual_decisions must be a JSON array")

        blockers_by_id = self._validate_blockers(blockers)
        referenced_blockers: set[str] = set()
        for stage_name, stage_record in stages.items():
            if not isinstance(stage_name, str):
                raise ManifestError("manifest stage names must be strings")
            try:
                self._validate_stage_name(stage_name)
            except ValueError as exc:
                raise ManifestError(str(exc)) from exc
            self._validate_stage_record(stage_name, stage_record, blockers_by_id)
            referenced_blockers.update(stage_record["blocker_ids"])
        for blocker_id, blocker in blockers_by_id.items():
            if blocker.get("resolved_at") is None and blocker_id not in referenced_blockers:
                raise ManifestError(
                    f"manifest has unresolved blocker {blocker_id!r} without a blocked stage"
                )
        self._validate_artifacts(artifacts, stages)
        self._validate_manual_decisions(decisions)

    def _validate_stage_record(
        self,
        stage_name: str,
        value: Any,
        blockers_by_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        stage = self._require_mapping(value, f"stages.{stage_name}")
        self._missing_fields(
            stage, _REQUIRED_STAGE_FIELDS, f"manifest stage {stage_name!r}"
        )
        try:
            status = StageStatus(stage["status"])
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"stage {stage_name!r} has an invalid status") from exc
        attempts = stage["attempts"]
        if type(attempts) is not int or attempts < 0:
            raise ManifestError(
                f"manifest stages.{stage_name}.attempts must be a non-negative integer"
            )
        self._require_utc_timestamp(
            stage["started_at"], f"stages.{stage_name}.started_at", nullable=True
        )
        self._require_utc_timestamp(
            stage["finished_at"], f"stages.{stage_name}.finished_at", nullable=True
        )
        stage_artifacts = self._require_mapping(
            stage["artifacts"], f"stages.{stage_name}.artifacts"
        )
        for artifact_name, path in stage_artifacts.items():
            if not isinstance(artifact_name, str) or not artifact_name:
                raise ManifestError("manifest artifact names must be non-empty strings")
            if not isinstance(path, str):
                raise ManifestError(
                    f"manifest stages.{stage_name}.artifacts.{artifact_name} "
                    "must be a string"
                )
            try:
                self._safe_relative_path(path)
            except (TypeError, ValueError) as exc:
                raise ManifestError(
                    f"manifest artifact path {path!r} is unsafe: {exc}"
                ) from exc
        artifact_hashes = self._require_mapping(
            stage["artifact_hashes"], f"stages.{stage_name}.artifact_hashes"
        )
        if any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in artifact_hashes.items()
        ):
            raise ManifestError(
                f"manifest stages.{stage_name}.artifact_hashes must map strings to strings"
            )
        unknown_hashes = artifact_hashes.keys() - stage_artifacts.keys()
        if unknown_hashes:
            raise ManifestError(
                f"manifest stage {stage_name!r} has hashes without artifacts"
            )
        self._require_mapping(
            stage["validation"], f"stages.{stage_name}.validation"
        )
        blocker_ids = stage["blocker_ids"]
        if not isinstance(blocker_ids, list) or any(
            not isinstance(blocker_id, str) or not blocker_id
            for blocker_id in blocker_ids
        ):
            raise ManifestError(
                f"manifest stages.{stage_name}.blocker_ids must be a string array"
            )
        for blocker_id in blocker_ids:
            blocker = blockers_by_id.get(blocker_id)
            if blocker is None or blocker["stage"] != stage_name:
                raise ManifestError(
                    f"manifest stage {stage_name!r} references unknown blocker {blocker_id!r}"
                )
            if blocker.get("resolved_at") is not None:
                raise ManifestError(
                    f"manifest stage {stage_name!r} references resolved blocker {blocker_id!r}"
                )
        if status is StageStatus.BLOCKED and not blocker_ids:
            raise ManifestError(f"blocked stage {stage_name!r} requires a blocker")
        if status is not StageStatus.BLOCKED and blocker_ids:
            raise ManifestError(
                f"non-blocked stage {stage_name!r} cannot retain blocker_ids"
            )
        if "error" in stage:
            error = self._require_mapping(
                stage["error"], f"stages.{stage_name}.error"
            )
            self._missing_fields(
                error,
                frozenset({"type", "message"}),
                f"manifest stage {stage_name!r} error",
            )
            self._require_string(error["type"], f"stages.{stage_name}.error.type")
            if not isinstance(error["message"], str):
                raise ManifestError(
                    f"manifest stages.{stage_name}.error.message must be a string"
                )
            if "details" in error:
                self._require_mapping(
                    error["details"], f"stages.{stage_name}.error.details"
                )

        started_at = stage["started_at"]
        finished_at = stage["finished_at"]
        if started_at is not None and finished_at is not None:
            if from_utc_iso(finished_at) < from_utc_iso(started_at):
                raise ManifestError(
                    f"manifest stage {stage_name!r} finished_at precedes started_at"
                )
        has_error = "error" in stage
        if status is StageStatus.PENDING:
            if (
                attempts != 0
                or started_at is not None
                or finished_at is not None
                or has_error
                or stage_artifacts
                or artifact_hashes
                or stage["validation"]
            ):
                raise ManifestError(
                    f"pending stage {stage_name!r} has inconsistent lifecycle state"
                )
        elif status is StageStatus.RUNNING:
            if (
                attempts < 1
                or started_at is None
                or finished_at is not None
                or has_error
            ):
                raise ManifestError(
                    f"running stage {stage_name!r} requires attempts and started_at only"
                )
        elif status is StageStatus.COMPLETED:
            if (
                attempts < 1
                or started_at is None
                or finished_at is None
                or has_error
            ):
                raise ManifestError(
                    f"completed stage {stage_name!r} requires attempts, started_at, "
                    "finished_at, and no error"
                )
        elif status is StageStatus.FAILED:
            if (
                attempts < 1
                or started_at is None
                or finished_at is None
                or not has_error
            ):
                raise ManifestError(
                    f"failed stage {stage_name!r} requires attempts, timestamps, and error"
                )
        elif status is StageStatus.BLOCKED and finished_at is None:
            raise ManifestError(
                f"blocked stage {stage_name!r} requires finished_at and blocker links"
            )

    def _validate_artifacts(
        self, artifacts: Mapping[str, Any], stages: Mapping[str, Any]
    ) -> None:
        for artifact_name, value in artifacts.items():
            if not isinstance(artifact_name, str) or not artifact_name:
                raise ManifestError("manifest artifact names must be non-empty strings")
            artifact = self._require_mapping(value, f"artifacts.{artifact_name}")
            self._missing_fields(
                artifact,
                frozenset({"path", "stage"}),
                f"manifest artifact {artifact_name!r}",
            )
            self._require_string(
                artifact["path"], f"artifacts.{artifact_name}.path"
            )
            try:
                self._safe_relative_path(artifact["path"])
            except (TypeError, ValueError) as exc:
                raise ManifestError(
                    f"manifest artifact path {artifact['path']!r} is unsafe: {exc}"
                ) from exc
            self._require_string(
                artifact["stage"], f"artifacts.{artifact_name}.stage"
            )
            try:
                self._validate_stage_name(artifact["stage"])
            except ValueError as exc:
                raise ManifestError(str(exc)) from exc
            if "sha256" in artifact:
                self._require_string(
                    artifact["sha256"], f"artifacts.{artifact_name}.sha256"
                )
            if "validation" in artifact:
                self._require_mapping(
                    artifact["validation"], f"artifacts.{artifact_name}.validation"
                )
            owner = artifact["stage"]
            owner_stage = stages.get(owner)
            if not isinstance(owner_stage, Mapping):
                raise ManifestError(
                    f"manifest artifact {artifact_name!r} has unknown owner stage"
                )
            if owner_stage["artifacts"].get(artifact_name) != artifact["path"]:
                raise ManifestError(
                    f"manifest artifact {artifact_name!r} disagrees with owner stage"
                )
            owner_hash = owner_stage["artifact_hashes"].get(artifact_name)
            if artifact.get("sha256") != owner_hash:
                raise ManifestError(
                    f"manifest artifact {artifact_name!r} hash disagrees with owner stage"
                )
            owner_validation = owner_stage["validation"].get(artifact_name)
            if artifact.get("validation") != owner_validation:
                raise ManifestError(
                    f"manifest artifact {artifact_name!r} validation disagrees with owner stage"
                )

        for stage_name, stage in stages.items():
            for artifact_name, artifact_path in stage["artifacts"].items():
                global_artifact = artifacts.get(artifact_name)
                if not isinstance(global_artifact, Mapping):
                    raise ManifestError(
                        f"manifest stage artifact {artifact_name!r} has no global record"
                    )
                if (
                    global_artifact.get("stage") != stage_name
                    or global_artifact.get("path") != artifact_path
                ):
                    raise ManifestError(
                        f"manifest stage artifact {artifact_name!r} has inconsistent ownership"
                    )

    def _validate_blockers(
        self, blockers: list[Any]
    ) -> dict[str, Mapping[str, Any]]:
        blockers_by_id: dict[str, Mapping[str, Any]] = {}
        required = frozenset({"id", "stage", "code", "message", "created_at"})
        for index, value in enumerate(blockers):
            blocker = self._require_mapping(value, f"blockers[{index}]")
            self._missing_fields(
                blocker, required, f"manifest blocker at index {index}"
            )
            for field in required:
                self._require_string(blocker[field], f"blockers[{index}].{field}")
            try:
                self._validate_stage_name(blocker["stage"])
            except ValueError as exc:
                raise ManifestError(str(exc)) from exc
            blocker_id = blocker["id"]
            if blocker_id in blockers_by_id:
                raise ManifestError(f"duplicate manifest blocker id {blocker_id!r}")
            if "details" in blocker:
                self._require_mapping(blocker["details"], f"blockers[{index}].details")
            if "resolved_at" in blocker:
                self._require_utc_timestamp(
                    blocker["resolved_at"],
                    f"blockers[{index}].resolved_at",
                    nullable=True,
                )
            if "resolution" in blocker:
                self._require_string(
                    blocker["resolution"], f"blockers[{index}].resolution"
                )
            self._require_utc_timestamp(
                blocker["created_at"], f"blockers[{index}].created_at"
            )
            blockers_by_id[blocker_id] = blocker
        return blockers_by_id

    def _validate_manual_decisions(self, decisions: list[Any]) -> None:
        required = frozenset({"id", "decision", "actor", "recorded_at", "details"})
        for index, value in enumerate(decisions):
            decision = self._require_mapping(value, f"manual_decisions[{index}]")
            self._missing_fields(
                decision, required, f"manifest manual decision at index {index}"
            )
            for field in ("id", "decision", "actor", "recorded_at"):
                self._require_string(
                    decision[field], f"manual_decisions[{index}].{field}"
                )
            self._require_utc_timestamp(
                decision["recorded_at"], f"manual_decisions[{index}].recorded_at"
            )
            self._require_mapping(
                decision["details"], f"manual_decisions[{index}].details"
            )

    def reload(self) -> None:
        with self._exclusive_lock():
            self._data = self._load()

    def _write_atomic(self, document: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    document,
                    temporary,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
            directory_fd = os.open(
                self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _commit(self, candidate: Mapping[str, Any]) -> None:
        safe_candidate = redact_value(copy.deepcopy(dict(candidate)))
        if not isinstance(safe_candidate, dict):
            raise ManifestError("manifest root must remain a JSON object")
        self._validate_document(safe_candidate)
        self._write_atomic(safe_candidate)
        self._data = safe_candidate

    @staticmethod
    def _validate_stage_name(stage: str) -> None:
        if not isinstance(stage, str) or not _STAGE_NAME_RE.fullmatch(stage):
            raise ValueError(f"invalid stage name: {stage!r}")

    def _safe_relative_path(self, value: str | os.PathLike[str]) -> str:
        raw = os.fspath(value)
        if not isinstance(raw, str):
            raise ManifestPathError("managed paths must be text paths")
        if not raw or "\x00" in raw or "\\" in raw:
            raise ManifestPathError("managed path is not a safe relative path")
        path = Path(raw)
        if ".." in path.parts:
            raise ManifestPathError("managed path may not contain '..'")
        candidate = path if path.is_absolute() else self.run_dir / path
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(self.run_dir)
        except ValueError as exc:
            raise ManifestPathError("managed path escapes the run directory") from exc
        if relative == Path("."):
            raise ManifestPathError("managed path must name a file or subdirectory")
        return relative.as_posix()

    def _status(self, document: Mapping[str, Any], stage: str) -> StageStatus:
        raw = document.get("stages", {}).get(stage, {}).get(
            "status", StageStatus.PENDING.value
        )
        try:
            return StageStatus(raw)
        except ValueError as exc:
            raise ManifestError(f"stage {stage!r} has an invalid status") from exc

    def stage(self, stage: str) -> dict[str, Any]:
        self._validate_stage_name(stage)
        with self._exclusive_lock():
            self._data = self._load()
            record = self._data.get("stages", {}).get(stage)
            if record is None:
                return {
                    "status": StageStatus.PENDING.value,
                    "attempts": 0,
                    "started_at": None,
                    "finished_at": None,
                    "artifacts": {},
                    "artifact_hashes": {},
                    "validation": {},
                    "blocker_ids": [],
                }
            return copy.deepcopy(record)

    def should_run(self, stage: str, *, force: bool = False) -> bool:
        self._validate_stage_name(stage)
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        if force:
            return True
        with self._exclusive_lock():
            self._data = self._load()
            return self._status(self._data, stage) in {
                StageStatus.PENDING,
                StageStatus.RUNNING,
                StageStatus.FAILED,
            }

    def start(self, stage: str, *, force: bool = False) -> None:
        with self._exclusive_lock():
            self._data = self._load()
            self._start_current(stage, force=force)

    def _start_current(self, stage: str, *, force: bool = False) -> None:
        self._validate_stage_name(stage)
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        current = self._status(self._data, stage)
        if current in {StageStatus.COMPLETED, StageStatus.BLOCKED} and not force:
            raise ManifestTransitionError(
                f"stage {stage!r} is {current.value}; explicit force is required"
            )
        if current not in {
            StageStatus.PENDING,
            StageStatus.RUNNING,
            StageStatus.FAILED,
            StageStatus.BLOCKED,
            StageStatus.COMPLETED,
        }:
            raise ManifestTransitionError(
                f"stage {stage!r} cannot start from {current.value!r}"
            )

        candidate = copy.deepcopy(self._data)
        previous = candidate.setdefault("stages", {}).get(stage, {})
        transition_at = utc_now_iso()
        top_level_artifacts = candidate.setdefault("artifacts", {})
        for artifact_name, artifact in list(top_level_artifacts.items()):
            if isinstance(artifact, Mapping) and artifact.get("stage") == stage:
                del top_level_artifacts[artifact_name]
        if current is StageStatus.BLOCKED and force:
            linked_blockers = frozenset(previous.get("blocker_ids", []))
            for blocker in candidate.setdefault("blockers", []):
                if blocker.get("id") in linked_blockers:
                    blocker["resolved_at"] = transition_at
                    blocker["resolution"] = "forced_rerun"
        candidate["stages"][stage] = {
            "status": StageStatus.RUNNING.value,
            "attempts": int(previous.get("attempts", 0)) + 1,
            "started_at": transition_at,
            "finished_at": None,
            "artifacts": {},
            "artifact_hashes": {},
            "validation": {},
            "blocker_ids": [],
        }
        self._touch_and_commit(candidate)

    def complete(
        self,
        stage: str,
        *,
        artifacts: Mapping[str, str | os.PathLike[str]] | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        with self._exclusive_lock():
            self._data = self._load()
            self._complete_current(
                stage,
                artifacts=artifacts,
                artifact_hashes=artifact_hashes,
                validation=validation,
            )

    def _complete_current(
        self,
        stage: str,
        *,
        artifacts: Mapping[str, str | os.PathLike[str]] | None = None,
        artifact_hashes: Mapping[str, str] | None = None,
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        self._validate_stage_name(stage)
        if self._status(self._data, stage) is not StageStatus.RUNNING:
            raise ManifestTransitionError(
                f"stage {stage!r} can complete only from running"
            )

        safe_artifacts: dict[str, str] = {}
        for name, artifact_path in (artifacts or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("artifact names must be non-empty strings")
            safe_artifacts[name] = self._safe_relative_path(artifact_path)
        hashes = dict(artifact_hashes or {})
        unknown_hashes = hashes.keys() - safe_artifacts.keys()
        if unknown_hashes:
            raise ValueError(
                f"artifact hashes have no matching artifact: {sorted(unknown_hashes)!r}"
            )
        validation_data = dict(validation or {})

        for name in safe_artifacts:
            existing = self._data.get("artifacts", {}).get(name)
            if isinstance(existing, Mapping) and existing.get("stage") != stage:
                raise ManifestTransitionError(
                    f"artifact {name!r} is already owned by stage "
                    f"{existing.get('stage')!r}"
                )

        candidate = copy.deepcopy(self._data)
        record = candidate["stages"][stage]
        record.update(
            {
                "status": StageStatus.COMPLETED.value,
                "finished_at": utc_now_iso(),
                "artifacts": safe_artifacts,
                "artifact_hashes": hashes,
                "validation": validation_data,
            }
        )
        record.pop("error", None)
        for name, artifact_path in safe_artifacts.items():
            artifact_record: dict[str, Any] = {
                "path": artifact_path,
                "stage": stage,
            }
            if name in hashes:
                artifact_record["sha256"] = hashes[name]
            if name in validation_data:
                artifact_record["validation"] = validation_data[name]
            candidate.setdefault("artifacts", {})[name] = artifact_record
        self._touch_and_commit(candidate)

    def fail(self, stage: str, error: BaseException | str | Mapping[str, Any]) -> None:
        with self._exclusive_lock():
            self._data = self._load()
            self._fail_current(stage, error)

    def _fail_current(
        self, stage: str, error: BaseException | str | Mapping[str, Any]
    ) -> None:
        self._validate_stage_name(stage)
        if self._status(self._data, stage) is not StageStatus.RUNNING:
            raise ManifestTransitionError(f"stage {stage!r} can fail only from running")

        if isinstance(error, BaseException):
            error_type = type(error).__name__
            safe_error = redact_value(error)
            safe_message = str(safe_error).removeprefix(f"{error_type}: ")
            summary = ErrorSummary(error_type, safe_message).to_dict()
        elif isinstance(error, str):
            summary = ErrorSummary("Error", error).to_dict()
        elif isinstance(error, Mapping):
            summary = {
                "type": str(error.get("type", "Error")),
                "message": str(error.get("message", "stage failed")),
            }
            details = error.get("details")
            if isinstance(details, Mapping) and details:
                summary["details"] = dict(details)
        else:
            raise TypeError("error must be an exception, string, or error mapping")

        candidate = copy.deepcopy(self._data)
        candidate["stages"][stage].update(
            {
                "status": StageStatus.FAILED.value,
                "finished_at": utc_now_iso(),
                "error": summary,
            }
        )
        self._touch_and_commit(candidate)

    def block(
        self,
        stage: str,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        with self._exclusive_lock():
            self._data = self._load()
            return self._block_current(
                stage, code=code, message=message, details=details
            )

    def _block_current(
        self,
        stage: str,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        self._validate_stage_name(stage)
        current = self._status(self._data, stage)
        if current is StageStatus.COMPLETED:
            raise ManifestTransitionError(f"completed stage {stage!r} cannot be blocked")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("blocker code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("blocker message must be a non-empty string")

        blocker_id = uuid.uuid4().hex
        blocker = BlockerRecord(
            id=blocker_id,
            stage=stage,
            code=code,
            message=message,
            created_at=utc_now_iso(),
            details=dict(details or {}),
        ).to_dict()
        candidate = copy.deepcopy(self._data)
        stages = candidate.setdefault("stages", {})
        previous = stages.get(stage, {})
        blocker_ids = [*previous.get("blocker_ids", []), blocker_id]
        stages[stage] = {
            **previous,
            "status": StageStatus.BLOCKED.value,
            "attempts": int(previous.get("attempts", 0)),
            "started_at": previous.get("started_at"),
            "finished_at": utc_now_iso(),
            "artifacts": dict(previous.get("artifacts", {})),
            "artifact_hashes": dict(previous.get("artifact_hashes", {})),
            "validation": dict(previous.get("validation", {})),
            "blocker_ids": blocker_ids,
        }
        stages[stage].pop("error", None)
        candidate.setdefault("blockers", []).append(blocker)
        self._touch_and_commit(candidate)
        return blocker_id

    def record_manual_decision(
        self,
        decision_id: str,
        decision: str,
        *,
        actor: str = "operator",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self._exclusive_lock():
            self._data = self._load()
            self._record_manual_decision_current(
                decision_id, decision, actor=actor, details=details
            )

    def _record_manual_decision_current(
        self,
        decision_id: str,
        decision: str,
        *,
        actor: str = "operator",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("decision_id must be a non-empty string")
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError("decision must be a non-empty string")
        candidate = copy.deepcopy(self._data)
        candidate.setdefault("manual_decisions", []).append(
            {
                "id": decision_id,
                "decision": decision,
                "actor": actor,
                "recorded_at": utc_now_iso(),
                "details": dict(details or {}),
            }
        )
        self._touch_and_commit(candidate)

    def update_metadata(
        self,
        *,
        media_metadata: Mapping[str, Any] | None = None,
        inferred_event: Mapping[str, Any] | None = None,
        tool_versions: Mapping[str, Any] | None = None,
        model_versions: Mapping[str, Any] | None = None,
    ) -> None:
        updates = {
            "media_metadata": media_metadata,
            "inferred_event": inferred_event,
            "tool_versions": tool_versions,
            "model_versions": model_versions,
        }
        if all(value is None for value in updates.values()):
            raise ValueError("at least one metadata update is required")
        for name, value in updates.items():
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")

        with self._exclusive_lock():
            self._data = self._load()
            candidate = copy.deepcopy(self._data)
            if media_metadata is not None:
                candidate["input"]["media"].update(dict(media_metadata))
            for field, value in (
                ("inferred_event", inferred_event),
                ("tool_versions", tool_versions),
                ("model_versions", model_versions),
            ):
                if value is not None:
                    candidate[field].update(dict(value))
            self._touch_and_commit(candidate)

    def replace_inferred_event(self, inferred_event: Mapping[str, Any]) -> None:
        """Replace (not merge) the resolved event, e.g. clear it on a re-run."""

        if not isinstance(inferred_event, Mapping):
            raise TypeError("inferred_event must be a mapping")
        with self._exclusive_lock():
            self._data = self._load()
            candidate = copy.deepcopy(self._data)
            candidate["inferred_event"] = dict(inferred_event)
            self._touch_and_commit(candidate)

    def _touch_and_commit(self, candidate: dict[str, Any]) -> None:
        candidate["updated_at"] = utc_now_iso()
        self._commit(candidate)


RunManifest = Manifest
