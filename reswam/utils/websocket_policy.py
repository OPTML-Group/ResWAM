"""Websocket policy server/client and B-spline trajectory smoothing for evaluation."""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
import traceback
from typing import Any

import msgpack
import numpy as np
from scipy.interpolate import make_lsq_spline


def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: Any) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def bspline_smooth(action_seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    """Least-squares B-spline smoothing over an action chunk [N, D]."""
    action_seq = np.asarray(action_seq, dtype=np.float64)
    n_steps, _ = action_seq.shape
    if n_steps <= num_ctrl_pts:
        return action_seq.astype(np.float32)
    x = np.arange(n_steps)
    num_internal_knots = num_ctrl_pts - degree
    internal_knots = np.linspace(0, n_steps - 1, num_internal_knots + 2)[1:-1]
    knots = np.concatenate([[0] * (degree + 1), internal_knots, [n_steps - 1] * (degree + 1)])
    spline = make_lsq_spline(x, action_seq, knots, k=degree)
    return spline(x).astype(np.float32)


class WebsocketPolicyServer:
    """Thread-safe Websocket policy inference server for simulator clients."""

    def __init__(
        self,
        policy: Any,
        host: str = "127.0.0.1",
        port: int = 10093,
        idle_timeout: int = -1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = int(port)
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._policy_lock = threading.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server

        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server: Any) -> None:
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: Any) -> None:
        import websockets
        import websockets.frames

        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                msg = unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = await asyncio.to_thread(self._route_message, msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise

    def _route_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}
        if mtype in ("infer", "predict_action"):
            try:
                with self._policy_lock:
                    output_dict = self._policy.predict_action(**msg)
            except Exception as exc:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": str(exc)},
                }
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": output_dict,
            }
        return {
            "status": "error",
            "ok": False,
            "type": "unknown",
            "request_id": req_id,
            "error": {"message": f"Unsupported message type '{mtype}'"},
        }


class WebsocketClientPolicy:
    """Synchronous Websocket client for querying a WebsocketPolicyServer."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 10093,
        api_key: str | None = None,
        ping_interval: float | None = 30.0,
        ping_timeout: float | None = 180.0,
    ) -> None:
        self._uri = f"ws://{host}" + (f":{port}" if port is not None else "")
        self._packer = Packer()
        self._api_key = api_key
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _connect(self):
        import websockets.sync.client

        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        return websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=150,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )

    def _wait_for_server(self, timeout: float = 300.0):
        start_time = time.time()
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to {self._uri} within {timeout} seconds")
            try:
                conn = self._connect()
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                time.sleep(2.0)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    def predict_action(self, query_info: dict[str, Any]) -> dict[str, Any]:
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)
