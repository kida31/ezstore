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

"""Process-wide in-memory ADK services.

The workshop runs one agent process, so in-memory sessions are sufficient. The
same instance is registered under ``shared://`` so the ADK web routes and the
A2A route see each other's sessions.
"""

from __future__ import annotations

import functools

from google.adk.artifacts import InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.sessions import InMemorySessionService

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"


@functools.cache
def get_session_service() -> InMemorySessionService:
    return InMemorySessionService()


@functools.cache
def get_artifact_service() -> InMemoryArtifactService:
    return InMemoryArtifactService()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
