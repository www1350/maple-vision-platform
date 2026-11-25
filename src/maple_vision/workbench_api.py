"""Optional FastAPI interface for the local offline lab."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .input_bindings import InputBindingProfile
from .lab import FAULTS
from .sim_types import SimAction
from .workbench import Conflict, Workbench

Identifier = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Name = Annotated[str, Field(min_length=1, max_length=80)]
Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]


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


def create_app(root: Path, examples: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = Workbench(root, examples)
        app.state.manager = manager
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
            if length is None or not length.isdecimal() or len(length) > 6 or int(length) > 4096:
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
            identity, "key-step", config.revision, key=config.key, bindings=config.bindings
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
                    "SELECT revision, sequence, key_name, skill, accepted, reason, events "
                    "FROM key_steps WHERE sim_id=? ORDER BY revision DESC LIMIT 50",
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
