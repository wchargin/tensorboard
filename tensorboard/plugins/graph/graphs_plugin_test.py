# -*- coding: utf-8 -*-
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for the Graphs Plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os.path

import tensorflow as tf

from google.protobuf import text_format
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer,
)  # pylint: disable=line-too-long
from tensorboard.compat.proto import config_pb2
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.util import test_util

tf.compat.v1.disable_v2_behavior()


# TODO(stephanwlee): Move more tests into the base class when v2 test
# can write graph and metadata with a TF public API.


class GraphsPluginBaseTest(object):

    _RUN_WITH_GRAPH = "_RUN_WITH_GRAPH"
    _RUN_WITHOUT_GRAPH = "_RUN_WITHOUT_GRAPH"

    _METADATA_TAG = "secret-stats"
    _MESSAGE_PREFIX_LENGTH_LOWER_BOUND = 1024

    def __init__(self, *args, **kwargs):
        super(GraphsPluginBaseTest, self).__init__(*args, **kwargs)
        self.logdir = None
        self.plugin = None

    def setUp(self):
        super(GraphsPluginBaseTest, self).setUp()
        self.logdir = self.get_temp_dir()

    def generate_run(self, run_name, include_graph, include_run_metadata):
        """Create a run."""
        raise NotImplementedError("Please implement generate_run")

    def set_up_with_runs(self, with_graph=True, without_graph=True):
        if with_graph:
            self.generate_run(
                self._RUN_WITH_GRAPH,
                include_graph=True,
                include_run_metadata=True,
            )
        if without_graph:
            self.generate_run(
                self._RUN_WITHOUT_GRAPH,
                include_graph=False,
                include_run_metadata=True,
            )
        self.bootstrap_plugin()

    def bootstrap_plugin(self):
        multiplexer = event_multiplexer.EventMultiplexer()
        multiplexer.AddRunsFromDirectory(self.logdir)
        multiplexer.Reload()
        context = base_plugin.TBContext(
            logdir=self.logdir, multiplexer=multiplexer
        )
        self.plugin = graphs_plugin.GraphsPlugin(context)

    def testRoutesProvided(self):
        """Tests that the plugin offers the correct routes."""
        self.set_up_with_runs()
        routes = self.plugin.get_plugin_apps()
        self.assertIsInstance(routes["/graph"], collections.Callable)
        self.assertIsInstance(routes["/run_metadata"], collections.Callable)
        self.assertIsInstance(routes["/info"], collections.Callable)


class GraphsPluginV1Test(GraphsPluginBaseTest, tf.test.TestCase):
    def generate_run(self, run_name, include_graph, include_run_metadata):
        """Create a run with a text summary, metadata, and optionally a
        graph."""
        tf.compat.v1.reset_default_graph()
        k1 = tf.constant(math.pi, name="k1")
        k2 = tf.constant(math.e, name="k2")
        result = (k1 ** k2) - k1
        expected = tf.constant(20.0, name="expected")
        error = tf.abs(result - expected, name="error")
        message_prefix_value = "error " * 1000
        true_length = len(message_prefix_value)
        assert (
            true_length > self._MESSAGE_PREFIX_LENGTH_LOWER_BOUND
        ), true_length
        message_prefix = tf.constant(
            message_prefix_value, name="message_prefix"
        )
        error_message = tf.strings.join(
            [message_prefix, tf.as_string(error, name="error_string")],
            name="error_message",
        )
        summary_message = tf.compat.v1.summary.text(
            "summary_message", error_message
        )

        sess = tf.compat.v1.Session()
        writer = test_util.FileWriter(os.path.join(self.logdir, run_name))
        if include_graph:
            writer.add_graph(sess.graph)
        options = tf.compat.v1.RunOptions(
            trace_level=tf.compat.v1.RunOptions.FULL_TRACE
        )
        run_metadata = config_pb2.RunMetadata()
        s = sess.run(
            summary_message, options=options, run_metadata=run_metadata
        )
        writer.add_summary(s)
        if include_run_metadata:
            writer.add_run_metadata(run_metadata, self._METADATA_TAG)
        writer.close()

    def _get_graph(self, *args, **kwargs):
        """Set up runs, then fetch and return the graph as a proto."""
        self.set_up_with_runs()
        (graph_pbtxt, mime_type) = self.plugin.graph_impl(
            self._RUN_WITH_GRAPH, *args, **kwargs
        )
        self.assertEqual(mime_type, "text/x-protobuf")
        return text_format.Parse(graph_pbtxt, tf.compat.v1.GraphDef())

    def test_info(self):
        expected = {
            "w_graph_w_meta": {
                "run": "w_graph_w_meta",
                "run_graph": True,
                "tags": {
                    "secret-stats": {
                        "conceptual_graph": False,
                        "profile": True,
                        "tag": "secret-stats",
                        "op_graph": False,
                    },
                },
            },
            "w_graph_wo_meta": {
                "run": "w_graph_wo_meta",
                "run_graph": True,
                "tags": {},
            },
            "wo_graph_w_meta": {
                "run": "wo_graph_w_meta",
                "run_graph": False,
                "tags": {
                    "secret-stats": {
                        "conceptual_graph": False,
                        "profile": True,
                        "tag": "secret-stats",
                        "op_graph": False,
                    },
                },
            },
        }

        self.generate_run(
            "w_graph_w_meta", include_graph=True, include_run_metadata=True
        )
        self.generate_run(
            "w_graph_wo_meta", include_graph=True, include_run_metadata=False
        )
        self.generate_run(
            "wo_graph_w_meta", include_graph=False, include_run_metadata=True
        )
        self.generate_run(
            "wo_graph_wo_meta", include_graph=False, include_run_metadata=False
        )
        self.bootstrap_plugin()

        self.assertItemsEqual(expected, self.plugin.info_impl())

    def test_graph_simple(self):
        graph = self._get_graph(tag=None, is_conceptual=False)
        node_names = set(node.name for node in graph.node)
        self.assertEqual(
            {
                "k1",
                "k2",
                "pow",
                "sub",
                "expected",
                "sub_1",
                "error",
                "message_prefix",
                "error_string",
                "error_message",
                "summary_message",
                "summary_message/tag",
                "summary_message/serialized_summary_metadata",
            },
            node_names,
        )

    def test_graph_large_attrs(self):
        key = "o---;;-;"
        graph = self._get_graph(
            tag=None,
            is_conceptual=False,
            limit_attr_size=self._MESSAGE_PREFIX_LENGTH_LOWER_BOUND,
            large_attrs_key=key,
        )
        large_attrs = {
            node.name: list(node.attr[key].list.s)
            for node in graph.node
            if key in node.attr
        }
        self.assertEqual({"message_prefix": [b"value"]}, large_attrs)

    def test_run_metadata(self):
        self.set_up_with_runs()
        (metadata_pbtxt, mime_type) = self.plugin.run_metadata_impl(
            self._RUN_WITH_GRAPH, self._METADATA_TAG
        )
        self.assertEqual(mime_type, "text/x-protobuf")
        text_format.Parse(metadata_pbtxt, config_pb2.RunMetadata())
        # If it parses, we're happy.

    def test_is_active_with_graph_without_run_metadata(self):
        self.generate_run(
            "w_graph_wo_meta", include_graph=True, include_run_metadata=False
        )
        self.bootstrap_plugin()
        self.assertTrue(self.plugin.is_active())

    def test_is_active_without_graph_with_run_metadata(self):
        self.generate_run(
            "wo_graph_w_meta", include_graph=False, include_run_metadata=True
        )
        self.bootstrap_plugin()
        self.assertTrue(self.plugin.is_active())

    def test_is_active_with_both(self):
        self.generate_run(
            "w_graph_w_meta", include_graph=True, include_run_metadata=True
        )
        self.bootstrap_plugin()
        self.assertTrue(self.plugin.is_active())

    def test_is_active_without_both(self):
        self.generate_run(
            "wo_graph_wo_meta", include_graph=False, include_run_metadata=False
        )
        self.bootstrap_plugin()
        self.assertFalse(self.plugin.is_active())


if __name__ == "__main__":
    tf.test.main()
