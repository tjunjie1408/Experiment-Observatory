"""FastAPI application for the local-only M4 run service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse

from observatory.api.models import RunRequest
from observatory.api.service import (
    EventCursorError,
    RunConflictError,
    RunNotFoundError,
    RunService,
    WorkerStartError,
    WorkerTarget,
    _training_worker,
)


def create_app(
    *,
    runs_root: Path,
    repo_root: Path,
    worker_target: WorkerTarget = _training_worker,
    cancellation_timeout: float = 5.0,
) -> FastAPI:
    service = RunService(
        runs_root=runs_root,
        repo_root=repo_root,
        worker_target=worker_target,
        cancellation_timeout=cancellation_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        service.recover_orphans()
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="Experiment Observatory local run service", lifespan=lifespan)
    app.state.run_service = service

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    def start_run(request: RunRequest) -> dict[str, object]:
        try:
            return service.start(request).model_dump(by_alias=True)
        except RunConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except WorkerStartError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        try:
            return service.get(run_id).model_dump(by_alias=True)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
            ) from exc

    @app.post("/api/runs/{run_id}/cancel", response_model=None)
    def cancel_run(run_id: str) -> JSONResponse:
        try:
            manifest, accepted = service.cancel(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
            ) from exc
        if accepted:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=manifest.model_dump(by_alias=True),
            )
        return JSONResponse(content=manifest.model_dump(by_alias=True))

    @app.get("/api/runs/{run_id}/events")
    def events(
        run_id: str,
        after_seq: int | None = Query(default=None, ge=0),
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID", ge=0),
    ) -> StreamingResponse:
        if last_event_id is not None and after_seq is not None and last_event_id != after_seq:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Last-Event-ID and after_seq must match when both are supplied",
            )
        cursor = last_event_id if last_event_id is not None else (after_seq or 0)
        try:
            service.validate_event_cursor(run_id, cursor)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
            ) from exc
        except EventCursorError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return StreamingResponse(
            service.stream_events(run_id, cursor),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def main() -> None:
    repo_root = Path.cwd()
    app = create_app(runs_root=repo_root / "runs", repo_root=repo_root)
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
