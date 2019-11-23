# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `tensorboard.util.grpc_util`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import hashlib
import threading

from concurrent import futures
import grpc
import six

from tensorboard.util import grpc_util
from tensorboard.util import grpc_util_test_pb2
from tensorboard.util import grpc_util_test_pb2_grpc
from tensorboard.util import test_util
from tensorboard import test as tb_test
from tensorboard import version


def make_request(nonce):
    return grpc_util_test_pb2.TestRpcRequest(nonce=nonce)


def make_response(nonce):
    return grpc_util_test_pb2.TestRpcResponse(nonce=nonce)


class TestGrpcServer(grpc_util_test_pb2_grpc.TestServiceServicer):
    """Helper for testing gRPC client logic with a dummy gRPC server."""

    def __init__(self, handler):
        super(TestGrpcServer, self).__init__()
        self._handler = handler

    def TestRpc(self, request, context):
        return self._handler(request, context)

    @contextlib.contextmanager
    def run(self):
        """Context manager to run the gRPC server and yield a client for it."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        grpc_util_test_pb2_grpc.add_TestServiceServicer_to_server(self, server)
        port = server.add_secure_port(
            "localhost:0", grpc.local_server_credentials()
        )

        def launch_server():
            server.start()
            server.wait_for_termination()

        thread = threading.Thread(target=launch_server, name="TestGrpcServer")
        thread.daemon = True
        thread.start()
        with grpc.secure_channel(
            "localhost:%d" % port, grpc.local_channel_credentials()
        ) as channel:
            yield grpc_util_test_pb2_grpc.TestServiceStub(channel)
        server.stop(grace=None)
        thread.join()


class CallWithRetriesTest(tb_test.TestCase):
    def test_call_with_retries_succeeds(self):
        def handler(request, _):
            return make_response(request.nonce)

        server = TestGrpcServer(handler)
        with server.run() as client:
            response = grpc_util.call_with_retries(
                client.TestRpc, make_request(42)
            )
        self.assertEqual(make_response(42), response)

    def test_call_with_retries_fails_immediately_on_permanent_error(self):
        def handler(_, context):
            context.abort(grpc.StatusCode.INTERNAL, "foo")

        server = TestGrpcServer(handler)
        with server.run() as client:
            with self.assertRaises(grpc.RpcError) as raised:
                grpc_util.call_with_retries(client.TestRpc, make_request(42))
        self.assertEqual(grpc.StatusCode.INTERNAL, raised.exception.code())
        self.assertEqual("foo", raised.exception.details())

    def test_call_with_retries_fails_after_backoff_on_nonpermanent_error(self):
        attempt_times = []
        fake_time = test_util.FakeTime()

        def handler(_, context):
            attempt_times.append(fake_time.time())
            context.abort(grpc.StatusCode.UNAVAILABLE, "foo")

        server = TestGrpcServer(handler)
        with server.run() as client:
            with self.assertRaises(grpc.RpcError) as raised:
                grpc_util.call_with_retries(
                    client.TestRpc, make_request(42), fake_time
                )
        self.assertEqual(grpc.StatusCode.UNAVAILABLE, raised.exception.code())
        self.assertEqual("foo", raised.exception.details())
        self.assertLen(attempt_times, 5)
        self.assertBetween(attempt_times[1] - attempt_times[0], 2, 4)
        self.assertBetween(attempt_times[2] - attempt_times[1], 4, 8)
        self.assertBetween(attempt_times[3] - attempt_times[2], 8, 16)
        self.assertBetween(attempt_times[4] - attempt_times[3], 16, 32)

    def test_call_with_retries_succeeds_after_backoff_on_transient_error(self):
        attempt_times = []
        fake_time = test_util.FakeTime()

        def handler(request, context):
            attempt_times.append(fake_time.time())
            if len(attempt_times) < 3:
                context.abort(grpc.StatusCode.UNAVAILABLE, "foo")
            return make_response(request.nonce)

        server = TestGrpcServer(handler)
        with server.run() as client:
            response = grpc_util.call_with_retries(
                client.TestRpc, make_request(42), fake_time
            )
        self.assertEqual(make_response(42), response)
        self.assertLen(attempt_times, 3)
        self.assertBetween(attempt_times[1] - attempt_times[0], 2, 4)
        self.assertBetween(attempt_times[2] - attempt_times[1], 4, 8)

    def test_call_with_retries_includes_version_metadata(self):
        def digest(s):
            """Hashes a string into a 32-bit integer."""
            return (
                int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)
                & 0xFFFFFFFF
            )

        def handler(request, context):
            metadata = context.invocation_metadata()
            client_version = grpc_util.extract_version(metadata)
            return make_response(digest(client_version))

        server = TestGrpcServer(handler)
        with server.run() as client:
            response = grpc_util.call_with_retries(
                client.TestRpc, make_request(0)
            )
        expected_nonce = digest(
            grpc_util.extract_version(grpc_util.version_metadata())
        )
        self.assertEqual(make_response(expected_nonce), response)


class VersionMetadataTest(tb_test.TestCase):
    def test_structure(self):
        result = grpc_util.version_metadata()
        self.assertIsInstance(result, tuple)
        for kv in result:
            self.assertIsInstance(kv, tuple)
            self.assertLen(kv, 2)
            (k, v) = kv
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, six.string_types)

    def test_roundtrip(self):
        result = grpc_util.extract_version(grpc_util.version_metadata())
        self.assertEqual(result, version.VERSION)


if __name__ == "__main__":
    tb_test.main()
