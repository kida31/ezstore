# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from collections.abc import AsyncIterator

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner

from quotation_agent.app_utils import services
from quotation_agent.app_utils.a2a import a2a_base_url, attach_a2a_routes
from quotation_agent.mcp_client import close as close_mcp

load_dotenv()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)
# This workshop runs locally / in Docker Compose, not on GCP. Do not try to
# export telemetry to Cloud Trace or import the GCP-only dependency set.
otel_to_cloud = False

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from quotation_agent.agent import app as adk_app
    from quotation_agent.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    base_url = a2a_base_url()
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
        app_url=base_url,
    )

    # Second A2A endpoint: the order agent, using its real graph. The
    # quotation agent's order_handoff node consumes this over A2A.
    from order_agent.agent import app as order_adk_app

    order_runner = Runner(
        app=order_adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    await attach_a2a_routes(
        app,
        agent=order_adk_app.root_agent,
        runner=order_runner,
        task_store=InMemoryTaskStore(),
        rpc_path="/a2a/order_agent",
        app_url=base_url,
    )
    try:
        yield
    finally:
        await close_mcp()


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "ezstore-agent"
app.description = "API for interacting with the EZStore quotation agent"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
