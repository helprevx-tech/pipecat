#
# Copyright (c) 2026, HelpRevX
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Streamable HTTP respects caller network policy and owns client cleanup."""

import asyncio
import importlib
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip("mcp")

from mcp import StdioServerParameters  # noqa: E402
from mcp.client.session_group import StreamableHttpParameters  # noqa: E402

from pipecat.services import mcp_service  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("family", ["httpx", "httpx2"])
async def test_custom_factory_receives_settings_and_closes_on_teardown(
    monkeypatch, failure, family
):
    http = importlib.import_module(family)
    clients = []
    received = []
    params = StreamableHttpParameters(
        url="https://tools.example/mcp",
        headers={"Authorization": "Bearer test"},
        timeout=7.0,
        sse_read_timeout=19.0,
        terminate_on_close=False,
    )

    def factory(**kwargs):
        received.append(kwargs)
        client = http.AsyncClient(**kwargs, follow_redirects=False, trust_env=False)
        clients.append(client)
        return client

    @asynccontextmanager
    async def transport(url, *, http_client, terminate_on_close):
        assert url == params.url
        assert http_client is clients[-1]
        assert not http_client.follow_redirects
        assert not terminate_on_close
        yield ("read", "write")

    transport.__module__ = mcp_service.streamable_http_client.__module__
    module = sys.modules[transport.__module__]
    monkeypatch.setattr(module, "httpx2", http if family == "httpx2" else None, raising=False)
    monkeypatch.setattr(module, "httpx", httpx, raising=False)
    monkeypatch.setattr(mcp_service, "streamable_http_client", transport)

    async def open_transport():
        async with mcp_service._streamable_http_transport(params, factory) as streams:
            assert streams == ("read", "write")
            assert not clients[-1].is_closed
            if failure:
                raise failure("interrupted")

    if failure:
        with pytest.raises(failure):
            await open_transport()
    else:
        await open_transport()
    assert clients[0].is_closed
    assert set(received[0]) == {"headers", "timeout"}
    assert received[0]["headers"] == params.headers
    assert received[0]["timeout"].connect == 7.0
    assert received[0]["timeout"].read == 19.0


@pytest.mark.asyncio
async def test_custom_factory_rejects_wrong_http_family_and_closes_client(monkeypatch):
    module = sys.modules[mcp_service.streamable_http_client.__module__]

    class OtherAsyncClient(httpx.AsyncClient):
        pass

    monkeypatch.setattr(
        module,
        "httpx2",
        SimpleNamespace(AsyncClient=OtherAsyncClient, Timeout=httpx.Timeout),
        raising=False,
    )
    client = httpx.AsyncClient()
    factory = MagicMock(return_value=client)
    with pytest.raises(TypeError, match="SDK's HTTP client family"):
        async with mcp_service._streamable_http_transport(
            StreamableHttpParameters(url="https://tools.example/mcp"), factory
        ):
            pytest.fail("Unsupported HTTP family was used")
    factory.assert_called_once()
    assert client.is_closed


def test_custom_factory_rejects_non_http_transports():
    with pytest.raises(TypeError, match="only supported for streamable HTTP"):
        mcp_service.MCPClient(
            StdioServerParameters(command="unused"), http_client_factory=MagicMock()
        )


@pytest.mark.asyncio
async def test_client_forwards_factory_for_each_owned_session(monkeypatch):
    factory = MagicMock()
    received = []

    @asynccontextmanager
    async def transport(params, *, http_client_factory):
        received.append(http_client_factory)
        yield (None, None)

    session = AsyncMock()
    session.__aenter__.return_value = session
    monkeypatch.setattr(mcp_service, "_streamable_http_transport", transport)
    monkeypatch.setattr(mcp_service, "ClientSession", lambda *args: session)
    client = mcp_service.MCPClient(
        StreamableHttpParameters(url="https://tools.example/mcp"), http_client_factory=factory
    )
    await client.start()
    await client.close()
    await client.start()
    await client.close()
    assert received == [factory, factory]
    assert session.initialize.await_count == 2


@pytest.mark.asyncio
async def test_default_factory_retains_sdk_redirect_behavior(monkeypatch):
    module = sys.modules[mcp_service.streamable_http_client.__module__]
    captured = []

    @asynccontextmanager
    async def transport(url, *, http_client, terminate_on_close):
        captured.append(http_client)
        assert http_client.follow_redirects
        yield ("read", "write")

    transport.__module__ = module.__name__
    monkeypatch.setattr(module, "httpx2", None, raising=False)
    monkeypatch.setattr(module, "httpx", httpx, raising=False)
    monkeypatch.setattr(mcp_service, "streamable_http_client", transport)
    async with mcp_service._streamable_http_transport(
        StreamableHttpParameters(url="https://tools.example/mcp")
    ):
        pass
    assert captured[0].is_closed
