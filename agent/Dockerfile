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

FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

WORKDIR /code

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY ./pyproject.toml ./README.md ./
COPY ./uv.lock ./
COPY ./quotation_agent ./quotation_agent
COPY ./order_agent ./order_agent

RUN uv sync --frozen --no-dev

ARG AGENT_VERSION=0.0.0
ENV AGENT_VERSION=${AGENT_VERSION}

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"

CMD ["uv", "run", "uvicorn", "quotation_agent.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080"]
