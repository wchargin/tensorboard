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
"""Utilities to add data class annotations to known summary types.

This should be effected after the `data_compat` transformation.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.plugins.histogram import metadata as histograms_metadata
from tensorboard.plugins.image import metadata as images_metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.util import tensor_util


def migrate_event(event):
    if event.HasField("graph_def"):
        return _migrate_graph_event(event)
    if event.HasField("summary"):
        return _migrate_summary_event(event)
    return (event,)


def _migrate_graph_event(old_event):
    result = event_pb2.Event()
    result.wall_time = old_event.wall_time
    result.step = old_event.step
    value = result.summary.value.add(tag=graphs_metadata.RUN_GRAPH_NAME)
    graph_bytes = old_event.graph_def
    value.tensor.CopyFrom(tensor_util.make_tensor_proto([graph_bytes]))
    value.metadata.plugin_data.plugin_name = graphs_metadata.PLUGIN_NAME
    value.metadata.plugin_data.content = b""  # DO NOT SUBMIT
    value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    # In the short term, keep both the old event and the new event to
    # maintain compatibility.
    return (old_event, result)


def _migrate_summary_event(event):
    old_values = event.summary.value
    new_values = [new for old in old_values for new in _migrate_value(old)]
    # Optimization: Don't create a new event if there were no changes.
    if len(old_values) == len(new_values) and all(
        x is y for (x, y) in zip(old_values, new_values)
    ):
        return (event,)
    result = event_pb2.Event()
    result.CopyFrom(event)
    del result.summary.value[:]
    result.summary.value.extend(new_values)
    return (result,)


def _migrate_value(value):
    """Convert an old value to a stream of new values."""
    if value.metadata.data_class != summary_pb2.DATA_CLASS_UNKNOWN:
        return (value,)
    transformer = {
        audio_metadata.PLUGIN_NAME: _migrate_audio_value,
        histograms_metadata.PLUGIN_NAME: _migrate_histogram_value,
        images_metadata.PLUGIN_NAME: _migrate_image_value,
        scalars_metadata.PLUGIN_NAME: _migrate_scalar_value,
    }.get(value.metadata.plugin_data.plugin_name, lambda v: (v,))
    return transformer(value)


def _migrate_scalar_value(value):
    # TODO(@wchargin): Bump `ScalarPluginData.version`, et al.?
    new_value = summary_pb2.Summary.Value()
    new_value.CopyFrom(value)
    new_value.metadata.data_class = summary_pb2.DATA_CLASS_SCALAR
    return (new_value,)


def _migrate_histogram_value(value):
    new_value = summary_pb2.Summary.Value()
    new_value.CopyFrom(value)
    new_value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    return (new_value,)


def _migrate_image_value(value):
    new_value = summary_pb2.Summary.Value()
    new_value.CopyFrom(value)
    new_value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    # NOTE(@wchargin): Not stripping width and height from tensor value.
    return (new_value,)


def _migrate_audio_value(value):
    # TODO(@wchargin): The audio tensor needs to be split into two
    # separate tensors: a length-`k` blob sequence containing the
    # encoded audio data, plus a shape-`[k]` tensor containing the
    # labels. It'll be easiest to make an atomic change that adds this
    # transformation and updates the audio plugin.
    return (value,)

    original_tensor = tensor_util.make_ndarray(value.tensor)

    audio_value = summary_pb2.Summary.Value()
    audio_value.tag = "audio/%s" % value.tag
    audio_value.metadata.CopyFrom(value.metadata)
    audio_value.metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    audio_tensor = original_tensor[:, 0]
    audio_value.tensor.CopyFrom(tensor_util.make_tensor_proto(audio_tensor))

    label_value = summary_pb2.Summary.Value()
    label_value.tag = "label/%s" % value.tag
    label_value.metadata.CopyFrom(value.metadata)
    label_value.metadata.data_class = summary_pb2.DATA_CLASS_TENSOR
    label_value.metadata.display_name = ""
    label_value.metadata.summary_description = ""
    label_tensor = original_tensor[:, 1]
    label_value.tensor.CopyFrom(tensor_util.make_tensor_proto(label_tensor))

    # TODO(@wchargin): Consider explicitly tagging and linking these two
    # summaries via `PluginData.content` additions, rather than relying on
    # the namespaces.
    return (audio_value, label_value)
