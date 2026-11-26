"""Optional FastAPI interface for the local offline lab."""

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .input_bindings import (
    MAX_INPUT_BINDINGS_BYTES,
    MAX_INPUT_SOURCE_IMAGE_BYTES,
    InputBindingProfile,
    confirm_input_profile,
)
from .lab import FAULTS
from .recording_evaluation import (
    _event_report,
    _recording_window_report,
    _verify_event_report,
    _verify_recording_window_report,
)
from .recordings import _load_frame, _validated_manifest
from .sim_types import SimAction
from .trial_evidence import verify_trial_evidence
from .workbench import Conflict, Workbench

Identifier = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
RecordingIdentifier = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=80)]
Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
MAX_WORKBENCH_RECORDINGS = 128
MAX_WORKBENCH_EVENT_REPORTS = 256
MAX_WORKBENCH_COVERAGE_REPORTS = 64


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SceneInput(StrictModel):
    job: Name
    map_id: Name = "builtin"
    seed: Seed = 0


class KeyBindingEntryInput(StrictModel):
    key: Name
    skill: Name


class InputProfileInput(StrictModel):
    schema_version: Literal[1]
    profile_id: Annotated[str, Field(min_length=1, max_length=100)]
    bindings: Annotated[list[KeyBindingEntryInput], Field(min_length=1, max_length=128)]


class InputSourceImageInput(StrictModel):
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    bytes: Annotated[int, Field(ge=1, le=MAX_INPUT_SOURCE_IMAGE_BYTES)]
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: Annotated[int, Field(ge=1, le=8192)]
    height: Annotated[int, Field(ge=1, le=8192)]

    @model_validator(mode="after")
    def check_pixels(self) -> Self:
        if self.width * self.height > 32 * 1024 * 1024:
            raise ValueError("source image exceeds the 32-megapixel limit")
        return self


class InputProfileConfirmationInput(StrictModel):
    schema_version: Literal[1]
    profile: InputProfileInput
    comparison_source: Literal["browser_selected_image_digest", "live_window_manual_comparison"]
    source_image: InputSourceImageInput | None
    user_confirmed: Literal[True]
    automatic_skill_recognition: Literal[False]

    @model_validator(mode="after")
    def check_source(self) -> Self:
        selected = self.comparison_source == "browser_selected_image_digest"
        if selected != (self.source_image is not None):
            raise ValueError("comparison source and selected-image metadata disagree")
        return self


class RunInput(SceneInput):
    kind: Literal["run"]
    max_steps: Annotated[int, Field(ge=1, le=2500)] = 1500
    model: Name = "pixel"
    fault: Name = "none"
    mission: Name = "standard"
    input_profile: InputProfileInput | None = None


class VerifyInput(StrictModel):
    kind: Literal["replay", "reinfer", "export_bundle", "verify_bundle"]
    source: Identifier


class ActionInput(StrictModel):
    kind: Name
    direction: int = 0
    skill: Name = "basic"
    target: Annotated[str, Field(max_length=80)] = ""


class KeyStepInput(StrictModel):
    profile_id: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    revision: Annotated[int, Field(ge=0)]
    key: Name
    bindings: Annotated[dict[Name, Name], Field(max_length=128)]


class ChangeInput(StrictModel):
    operation: Literal["step", "save", "reset", "restore"]
    revision: Annotated[int, Field(ge=0)]
    action: ActionInput | None = None

    @model_validator(mode="after")
    def check_action(self) -> Self:
        if (self.operation == "step") != (self.action is not None):
            raise ValueError("only step requests carry an action")
        return self


def _plain_directory(path: Path, label: str) -> Path:
    resolved = Path(path).absolute()
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{label} must be a real local directory")
    return resolved


def _recording_catalog(
    recordings: Path | None, event_reports: Path | None
) -> dict[str, dict[str, Any]]:
    if recordings is None:
        if event_reports is not None:
            raise ValueError("recording event reports require a recordings directory")
        return {}
    root = _plain_directory(recordings, "recordings")
    children = sorted(root.iterdir(), key=lambda path: path.name)
    if len(children) > MAX_WORKBENCH_RECORDINGS:
        raise ValueError("recordings directory exceeds the 128-session workbench limit")
    catalog: dict[str, dict[str, Any]] = {}
    for child in children:
        if child.is_symlink() or not child.is_dir():
            raise ValueError("recordings directory may contain only real session directories")
        manifest = _validated_manifest(child)
        identity = manifest["session_id"]
        if identity in catalog:
            raise ValueError("recordings directory contains duplicate session identities")
        catalog[identity] = {"root": child, "manifest": manifest, "events": []}
    if event_reports is None:
        return catalog
    report_root = _plain_directory(event_reports, "recording event reports")
    reports = sorted(report_root.iterdir(), key=lambda path: path.name)
    if len(reports) > MAX_WORKBENCH_EVENT_REPORTS:
        raise ValueError("recording event report directory exceeds the 256-file limit")
    for report_path in reports:
        if report_path.is_symlink() or not report_path.is_file() or report_path.suffix != ".json":
            raise ValueError("recording event report directory may contain only JSON files")
        report = _event_report(report_path)
        identity = report["source"].get("session_id")
        entry = catalog.get(identity)
        if entry is None:
            raise ValueError("recording event report references an unregistered session")
        verification = _verify_event_report(entry["manifest"], report_path)
        entry["events"].append({"report": report, "verification": verification})
    return catalog


def _recording_summary(entry: dict[str, Any], *, frames: bool = False) -> dict[str, Any]:
    manifest = entry["manifest"]
    summary = {
        **{
            key: manifest[key]
            for key in (
                "schema_version",
                "session_id",
                "scope",
                "source_kind",
                "frame_count",
                "fps",
                "timebase",
                "width",
                "height",
                "truncated",
                "completion_reason",
            )
        },
        **(
            {"source_start_frame": manifest["source_start_frame"]}
            if "source_start_frame" in manifest
            else {}
        ),
        "validated": True,
        "timing_accuracy": (
            "source_pts"
            if manifest["timebase"] == "video_presentation_ticks"
            else "nominal_fps_only"
        ),
        "source_name": manifest["source_name"],
        "source_sha256": manifest["source_sha256"],
        "permission": manifest["permission"],
        "license": manifest["license"],
        "event_reports": len(entry["events"]),
        "source_end_frame": manifest.get("source_start_frame", 0) + manifest["frame_count"] - 1,
    }
    if frames:
        summary["frames"] = [
            {
                "sequence": item["sequence"],
                "timestamp_ms": item["timestamp_ms"],
                "source_name": item["source_name"],
            }
            for item in manifest["frames"]
        ]
        summary["reviewed_event_reports"] = entry["events"]
        summary["limitations"] = [
            "fixed recording frames cannot validate counterfactual actions or action success",
            "visible event labels do not identify a skill, actor, target, or keyboard input",
            "synthetic mechanics, key bindings, and ammunition rules remain separate evidence",
        ]
    return summary


def _recording_coverages(
    recordings: dict[str, dict[str, Any]], reports: Sequence[Path] | None
) -> dict[str, dict[str, Any]]:
    if reports is None:
        return {}
    if not recordings:
        raise ValueError("recording coverage reports require registered recording sessions")
    if (
        not isinstance(reports, (list, tuple))
        or not 1 <= len(reports) <= MAX_WORKBENCH_COVERAGE_REPORTS
    ):
        raise ValueError("supply 1..64 recording coverage reports")
    catalog: dict[str, dict[str, Any]] = {}
    for report_path in map(Path, reports):
        if report_path.is_symlink() or not report_path.is_file() or report_path.suffix != ".json":
            raise ValueError("recording coverage reports must be regular local JSON files")
        report = _recording_window_report(report_path)
        sessions = []
        for row in report["windows"]:
            if not isinstance(row, dict) or not isinstance(row.get("session_id"), str):
                raise ValueError("invalid recording coverage window")
            session = recordings.get(row["session_id"])
            if session is None:
                raise ValueError("recording coverage report references an unregistered session")
            sessions.append(session)
        verification = _verify_recording_window_report(
            [session["manifest"] for session in sessions], report_path
        )
        identity = verification["coverage_sha256"]
        if not isinstance(identity, str) or identity in catalog:
            raise ValueError("duplicate recording coverage report")
        catalog[identity] = {
            "report": report,
            "verification": verification,
            "sessions": sessions,
        }
    return catalog


def _coverage_summary(entry: dict[str, Any], *, frames: bool = False) -> dict[str, Any]:
    report = entry["report"]
    sessions = entry["sessions"]
    reviewed_events = sum(
        len(item["report"]["events"]) for session in sessions for item in session["events"]
    )
    summary = {
        "coverage_sha256": report["coverage_sha256"],
        "source_name": report["source"]["source_name"],
        "source_sha256": report["source"]["source_sha256"],
        "permission": report["source"]["permission"],
        "license": report["source"]["license"],
        "window_count": report["window_count"],
        "decoded_frame_count": report["decoded_frame_count"],
        "decoded_source_range": report["decoded_source_range"],
        "contiguous": report["contiguous"],
        "decoder_stop_reached": report["decoder_stop_reached"],
        "timing_accuracy": "source_pts",
        "event_reports": sum(len(session["events"]) for session in sessions),
        "reviewed_events": reviewed_events,
        "verified": True,
    }
    if not frames:
        return summary
    clock = report["source"]["video_clock"]
    timebase = Fraction(clock["numerator"], clock["denominator"])
    first_pts = report["windows"][0]["first"]["pts"]
    frame_rows = []
    frame_by_source: dict[int, dict[str, Any]] = {}
    for window, session in zip(report["windows"], sessions, strict=True):
        manifest = session["manifest"]
        for item in manifest["frames"]:
            source_sequence = window["source_start_frame"] + item["sequence"]
            row = {
                "source_sequence": source_sequence,
                "sequence": item["sequence"],
                "session_id": manifest["session_id"],
                "source_name": item["source_name"],
                "pts": item["pts"],
                "timestamp_ms": round((item["pts"] - first_pts) * timebase * 1000),
            }
            frame_rows.append(row)
            frame_by_source[source_sequence] = row
    event_rows = []
    for window, session in zip(report["windows"], sessions, strict=True):
        offset = window["source_start_frame"]
        for event_report in session["events"]:
            for event in event_report["report"]["events"]:
                start = offset + event["start"]["sequence"]
                end = offset + event["end"]["sequence"]
                event_rows.append(
                    {
                        "kind": event["kind"],
                        "reviewer": event_report["report"]["reviewer"],
                        "verified": event_report["verification"]["verified"],
                        "start_source_sequence": start,
                        "end_source_sequence": end,
                        "start_timestamp_ms": frame_by_source[start]["timestamp_ms"],
                        "end_timestamp_ms": frame_by_source[end]["timestamp_ms"],
                    }
                )
    return {
        **summary,
        "windows": report["windows"],
        "frames": frame_rows,
        "reviewed_visible_events": sorted(
            event_rows,
            key=lambda event: (
                event["start_source_sequence"],
                event["end_source_sequence"],
                event["kind"],
            ),
        ),
        "limitations": report["limitations"]
        + [
            "fixed recording frames cannot validate counterfactual actions or action success",
            "synthetic mechanics, key bindings, and ammunition rules remain separate evidence",
        ],
    }


def create_app(
    root: Path,
    examples: Path | None = None,
    recordings: Path | None = None,
    recording_events: Path | None = None,
    recording_coverages: Sequence[Path] | None = None,
    trial_evidence_root: Path | None = None,
    trial_evidence_ledger: Path | None = None,
) -> FastAPI:
    if (trial_evidence_root is None) != (trial_evidence_ledger is None):
        raise ValueError("trial evidence root and ledger must be supplied together")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = Workbench(root, examples)
        app.state.manager = manager
        app.state.recordings = _recording_catalog(recordings, recording_events)
        app.state.recording_coverages = _recording_coverages(
            app.state.recordings, recording_coverages
        )
        app.state.trial_evidence = (
            None
            if trial_evidence_root is None or trial_evidence_ledger is None
            else verify_trial_evidence(trial_evidence_ledger, trial_evidence_root)
        )
        try:
            yield
        finally:
            manager.close()

    api = FastAPI(
        title="Maple Vision Workbench",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def manager() -> Workbench:
        service: Workbench = api.state.manager
        return service

    def recording(identity: str) -> dict[str, Any]:
        entry: dict[str, Any] | None = api.state.recordings.get(identity)
        if entry is None:
            raise HTTPException(404, "找不到指定的录像会话")
        return entry

    def recording_coverage(identity: str) -> dict[str, Any]:
        entry: dict[str, Any] | None = api.state.recording_coverages.get(identity)
        if entry is None:
            raise HTTPException(404, "找不到指定的录像覆盖报告")
        return entry

    @api.middleware("http")
    async def local_origin(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return Response(status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            return Response(status_code=403)
        if request.method == "POST":
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return Response(status_code=415)
            length = request.headers.get("content-length")
            if length is None or not length.isascii() or not length.isdecimal() or len(length) > 6:
                return Response(status_code=413)
            size = int(length)
            if size > 4096:
                path = request.url.path
                if path == "/api/input-profile":
                    limit = MAX_INPUT_BINDINGS_BYTES
                elif path in {"/api/jobs", "/api/input-profile/confirmation"} or re.fullmatch(
                    "/api/sessions/[0-9a-f]{32}/key-step", path
                ):
                    limit = MAX_INPUT_BINDINGS_BYTES + 4096
                else:
                    return Response(status_code=413)
                if size > limit:
                    return Response(status_code=413)
                if path == "/api/jobs":
                    try:
                        body = await request.json()
                    except (ValueError, RecursionError):
                        return Response(status_code=413)
                    if not isinstance(body, dict) or not isinstance(
                        body.get("input_profile"), dict
                    ):
                        return Response(status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @api.exception_handler(KeyError)
    async def missing(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"detail": "找不到指定的会话或证据"}, status_code=404)

    @api.exception_handler(ValueError)
    async def invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc)}, status_code=409 if isinstance(exc, Conflict) else 422
        )

    resources = files("maple_vision").joinpath("resources", "workbench")

    @api.get("/")
    def page() -> Response:
        return Response(resources.joinpath("index.html").read_bytes(), media_type="text/html")

    @api.get("/static/{name}")
    def asset(name: str) -> Response:
        types = {"app.css": "text/css", "app.js": "text/javascript"}
        if name not in types:
            raise HTTPException(404)
        return Response(resources.joinpath(name).read_bytes(), media_type=types[name])

    @api.get("/api/content")
    def content() -> dict[str, Any]:
        m = manager()
        return {
            "classes": {
                name: {
                    "skills": [] if package is None else [s.id for s in package.skills],
                    "version": "builtin" if package is None else package.version,
                }
                for name, package in m.classes.items()
            },
            "maps": {
                name: ["meadow", "cavern"] if pack is None else list(pack.map_ids)
                for name, pack in m.maps.items()
            },
            "world_graphs": {
                key: {"nodes": graph.nodes, "scope": graph.provenance.scope}
                for key, graph in m.world_graphs.items()
            },
            "mission_options": {name: m.compatible_missions(pack) for name, pack in m.maps.items()},
            "models": list(m.models),
            "missions": {k: None if v is None else v.to_dict() for k, v in m.missions.items()},
            "faults": list(FAULTS),
            "scope": "synthetic_fixture",
        }

    @api.get("/api/trial-evidence")
    def trial_evidence() -> dict[str, Any]:
        result: dict[str, Any] | None = api.state.trial_evidence
        return {"configured": False} if result is None else {"configured": True, **result}

    @api.get("/api/recordings")
    def recording_sessions() -> list[dict[str, Any]]:
        return [
            _recording_summary(entry)
            for _, entry in sorted(
                api.state.recordings.items(),
                key=lambda item: (
                    item[1]["manifest"].get("source_start_frame", 0),
                    item[1]["manifest"]["source_name"],
                    item[0],
                ),
            )
        ]

    @api.get("/api/recording-coverages")
    def recording_coverage_reports() -> list[dict[str, Any]]:
        return [
            _coverage_summary(entry)
            for _, entry in sorted(api.state.recording_coverages.items(), key=lambda item: item[0])
        ]

    @api.get("/api/recording-coverages/{identity}")
    def recording_coverage_detail(identity: RecordingIdentifier) -> dict[str, Any]:
        return _coverage_summary(recording_coverage(identity), frames=True)

    @api.get("/api/recording-coverages/{identity}/frames/{source_sequence}")
    def recording_coverage_frame(
        identity: RecordingIdentifier,
        source_sequence: Annotated[int, Field(ge=0, le=2**53)],
    ) -> Response:
        entry = recording_coverage(identity)
        for window, session in zip(entry["report"]["windows"], entry["sessions"], strict=True):
            if window["source_start_frame"] <= source_sequence <= window["source_end_frame"]:
                root_path: Path = session["root"]
                if root_path.is_symlink():
                    raise HTTPException(404, "找不到指定的录像帧")
                sequence = source_sequence - window["source_start_frame"]
                frame_value = _load_frame(root_path, session["manifest"]["frames"][sequence])
                return Response(frame_value.png_bytes(), media_type="image/png")
        raise HTTPException(404, "找不到指定的录像帧")

    @api.get("/api/recordings/{identity}")
    def recording_detail(identity: RecordingIdentifier) -> dict[str, Any]:
        return _recording_summary(recording(identity), frames=True)

    @api.get("/api/recordings/{identity}/frames/{sequence}")
    def recording_frame(
        identity: RecordingIdentifier,
        sequence: Annotated[int, Field(ge=0, le=9999)],
    ) -> Response:
        entry = recording(identity)
        root_path: Path = entry["root"]
        manifest = entry["manifest"]
        if root_path.is_symlink() or sequence >= manifest["frame_count"]:
            raise HTTPException(404, "找不到指定的录像帧")
        frame_value = _load_frame(root_path, manifest["frames"][sequence])
        return Response(frame_value.png_bytes(), media_type="image/png")

    @api.get("/api/world-graphs/{graph_id}/route")
    def route(graph_id: Name, source: Name, target: Name) -> dict[str, Any]:
        graph = manager().world_graphs.get(graph_id)
        if graph is None:
            raise HTTPException(404, "找不到指定的世界图")
        path = graph.shortest_path(source, target)
        return {
            "graph_sha256": graph.content_sha256,
            "scope": graph.provenance.scope,
            "route": path.to_dict() if path is not None else None,
        }

    @api.post("/api/input-profile")
    def input_profile(profile: InputProfileInput) -> dict[str, Any]:
        return InputBindingProfile.from_manifest(profile.model_dump()).manifest()

    @api.post("/api/input-profile/confirmation")
    def input_profile_confirmation(config: InputProfileConfirmationInput) -> dict[str, Any]:
        profile = InputBindingProfile.from_manifest(config.profile.model_dump())
        image = None if config.source_image is None else config.source_image.model_dump()
        return confirm_input_profile(profile, config.comparison_source, image)

    @api.get("/api/jobs")
    def jobs() -> list[dict[str, Any]]:
        return manager().entries()

    @api.post("/api/jobs", status_code=202)
    def submit(config: RunInput | VerifyInput) -> dict[str, str]:
        return {"id": manager().submit(config.kind, config.model_dump(exclude={"kind"}))}

    @api.get("/api/jobs/{identity}")
    def detail(identity: Identifier) -> dict[str, Any]:
        return manager().detail(identity)

    @api.post("/api/jobs/{identity}/cancel")
    def cancel(identity: Identifier) -> dict[str, str]:
        manager().cancel(identity)
        return {"state": "cancelled"}

    @api.get("/api/jobs/{identity}/frames/{name}")
    def frame(identity: Identifier, name: str) -> FileResponse:
        return FileResponse(manager().frame(identity, name), media_type="image/png")

    @api.get("/api/jobs/{identity}/trace")
    def trace(
        identity: Identifier, offset: Annotated[int, Query(ge=0, le=2500)] = 0
    ) -> list[dict[str, Any]]:
        return manager().trace(identity, offset)

    @api.post("/api/sessions", status_code=201)
    def new_session(config: SceneInput) -> dict[str, Any]:
        return manager().create_sim(config.job, config.map_id, config.seed)

    @api.post("/api/sessions/{identity}/key-step")
    def key_step(identity: Identifier, config: KeyStepInput) -> dict[str, Any]:
        return manager().session(
            identity,
            "key-step",
            config.revision,
            key=config.key,
            bindings=config.bindings,
            profile_id=config.profile_id,
        )

    @api.get("/api/sessions/{identity}/key-log")
    def key_log(identity: Identifier) -> list[dict[str, Any]]:
        with manager().db() as db:
            if db.execute("SELECT 1 FROM sims WHERE id=?", (identity,)).fetchone() is None:
                raise HTTPException(404, "找不到指定的模拟会话")
            return [
                {
                    **dict(row),
                    "accepted": bool(row["accepted"]),
                    "events": json.loads(row["events"]),
                }
                for row in db.execute(
                    "SELECT revision, sequence, key_name, skill, accepted, reason, events, "
                    "profile_id FROM key_steps WHERE sim_id=? ORDER BY revision DESC LIMIT 50",
                    (identity,),
                )
            ]

    @api.get("/api/sessions/{identity}")
    def read_session(identity: Identifier) -> dict[str, Any]:
        return manager().session(identity, "read")

    @api.post("/api/sessions/{identity}")
    def change(identity: Identifier, config: ChangeInput) -> dict[str, Any]:
        action = None if config.action is None else SimAction(**config.action.model_dump())
        return manager().session(identity, config.operation, config.revision, action)

    return api
