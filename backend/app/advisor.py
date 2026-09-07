"""Bridge to the financial-advisor Go/ADK service.

The frontend only ever talks to this FastAPI backend. This module forwards
session creation and chat turns to the separate Go service (see /agent) over
its ADK REST API (POST /apps/.../sessions, POST /run_sse), so the browser
never needs the agent service's URL or CORS configuration.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import AdvisorChatRequest, AdvisorSessionRequest

router = APIRouter(prefix="/api/advisor", tags=["advisor"])

APP_NAME = "financial_coordinator"


def _advisor_base_url() -> str:
    host = os.environ.get("FINANCIAL_ADVISOR_HOST")
    if host:
        return f"http://{host}"
    return os.environ.get("FINANCIAL_ADVISOR_URL", "http://localhost:8081")


@router.post("/session")
async def create_session(req: AdvisorSessionRequest):
    url = f"{_advisor_base_url()}/api/apps/{APP_NAME}/users/{req.user_id}/sessions"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json={})
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"advisor service unreachable: {e}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@router.post("/chat")
async def chat(req: AdvisorChatRequest):
    url = f"{_advisor_base_url()}/api/run_sse"
    payload = {
        "appName": APP_NAME,
        "userId": req.user_id,
        "sessionId": req.session_id,
        "newMessage": {"role": "user", "parts": [{"text": req.message}]},
        "streaming": True,
    }

    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise HTTPException(status_code=resp.status_code, detail=body.decode())
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.HTTPError as e:
                yield f"event: error\ndata: advisor service unreachable: {e}\n\n".encode()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
