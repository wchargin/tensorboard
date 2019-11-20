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
"""Debugging tests for local network stack on GitHub."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import socket
import unittest
from wsgiref import simple_server

from concurrent import futures
from werkzeug import wrappers
import requests


class NetworkTest(unittest.TestCase):
  """Tests for `fetch_server_info`."""

  def _start_server(self, app, use_nodename):
    """Starts a server and returns its origin ("http://localhost:PORT")."""
    (_, localhost) = _localhost()
    server_class = _make_ipv6_compatible_wsgi_server()
    server = simple_server.make_server(localhost, 0, app, server_class)
    executor = futures.ThreadPoolExecutor()
    future = executor.submit(server.serve_forever, poll_interval=0.01)

    def cleanup():
      server.shutdown()  # stop handling requests
      server.server_close()  # release port
      future.result(timeout=3)  # wait for server termination

    self.addCleanup(cleanup)
    if ":" in localhost:
      localhost = "[%s]" % localhost  # IPv6, presumably; HACK?
    if use_nodename:
      return "http://%s:%d" % (localhost, server.server_port)
    else:
      return "http://localhost:%d" % server.server_port

  def _test(self, use_nodename):
    @wrappers.BaseRequest.application
    def app(request):
      body = "Hello from %r!" % request.url
      return wrappers.BaseResponse(body)

    origin = self._start_server(app, use_nodename=use_nodename)
    print("Started server on %r" % origin)
    response = requests.get(origin)
    print("Got response: %r" % response)
    print("Response content:\n<<<\n%s\n>>>" % response.content.decode("utf-8"))

  def test_with_nodename(self):
    self._test(use_nodename=True)

  def test_with_localhost(self):
    self._test(use_nodename=False)


def _localhost():
  """Gets family and nodename for a loopback address."""
  s = socket
  infos = s.getaddrinfo(None, 0, s.AF_UNSPEC, s.SOCK_STREAM, 0, s.AI_ADDRCONFIG)
  (family, _, _, _, address) = infos[0]
  nodename = address[0]
  print("All infos: %r" % (infos,))
  print("Family: %r" % (family,))
  print("Nodename: %r" % (nodename,))
  print("---")
  return (family, nodename)


def _make_ipv6_compatible_wsgi_server():
  """Creates a `WSGIServer` subclass that works on IPv6-only machines."""
  address_family = _localhost()[0]
  attrs = {"address_family": address_family}
  bases = (simple_server.WSGIServer, object)  # `object` needed for py2
  return type("_Ipv6CompatibleWsgiServer", bases, attrs)


if __name__ == "__main__":
  unittest.main()
