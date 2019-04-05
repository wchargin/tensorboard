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
"""Public APIs for the HParams plugin.

Eager execution should be enabled (TODO(@wchargin): can we not?).

TODO(@wchargin): Docs. This module is under construction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import random
import time

from google.protobuf import struct_pb2
import six
import tensorflow as tf

from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary


class HParam(object):
  """A hyperparameter in an experiment.

  This class describes a hyperparameter in the abstract. It ranges over
  a domain of values, but is not bound to any particular value.
  """

  def __init__(self, name, domain, display_name=None, description=None):
    """Create a hyperparameter object.

    Args:
      name: A string ID for this hyperparameter, which should be unique
        within an experiment.
      domain: A `Domain` object describing the values that this
        hyperparameter can take on.
      display_name: An optional human-readable display name (`str`).
      description: An optional human-readable description (`str`).

    Raises:
      ValueError: If `domain` is not a `Domain`.
    """
    self._name = name
    self._domain = domain
    self._display_name = display_name
    self._description = description
    if not isinstance(self._domain, Domain):
      raise ValueError("not a domain: %r" % self._domain)

  def __str__(self):
    return "<HParam %r: %s>" % (self._name, self._domain)

  def __repr__(self):
    fields = [
        ("name", self._name),
        ("domain", self._domain),
        ("display_name", self._display_name),
        ("description", self._description),
    ]
    fields_string = ", ".join("%s=%r" % (k, v) for (k, v) in fields)
    return "HParam(%s)" % fields_string

  @property
  def name(self):
    return self._name

  @property
  def domain(self):
    return self._domain

  @property
  def display_name(self):
    return self._display_name

  @property
  def description(self):
    return self._description


@six.add_metaclass(abc.ABCMeta)
class Domain(object):
  """The domain of a hyperparameter.

  Domains are restricted to values of the simple types `float`, `int`,
  `str`, and `bool`.

  TODO(@wchargin): Seal this hierarchy? Provide catamorphisms? Factor
  out `sample_uniform`?
  """

  @abc.abstractproperty
  def dtype(self):
    """Data type of this domain: `float`, `int`, `str`, or `bool`."""
    pass

  @abc.abstractmethod
  def update_hparam_info(self, hparam_info):
    """Update an `HParamInfo` proto to include this domain.

    This should update the `type` field on the proto and exactly one of
    the `domain` variants on the proto.

    TODO(wchargin): This is ugly, but necessary because `Domain` is not
    reified in the API proto. Can we change that? (If not, at least this
    function is mostly an implementation detail.)

    Args:
      hparam_info: An `api_pb2.HParamInfo` proto to modify.
    """
    pass

  @abc.abstractmethod
  def sample_uniform(self, rng=random):
    """Sample a value from this domain uniformly at random.

    Args:
      rng: A `random.Random` object; defaults to the `random` module.

    Raises:
      IndexError: If the domain is empty.
    """
    pass


class IntInterval(Domain):
  """A domain that takes on all integer values in a closed interval."""

  def __init__(self, min_value=None, max_value=None):
    if not isinstance(min_value, int):
      raise ValueError("min_value must be an int: %r" % min_value)
    if not isinstance(max_value, int):
      raise ValueError("max_value must be an int: %r" % max_value)
    if min_value > max_value:
      raise ValueError("%r > %r" % (min_value, max_value))
    self._min_value = min_value
    self._max_value = max_value

  def __str__(self):
    return "[%s, %s]" % (self._min_value, self._max_value)

  def __repr__(self):
    return "IntInterval(%r, %r)" % (self._min_value, self._max_value)

  @property
  def dtype(self):
    return int

  @property
  def as_proto(self):
    pass

  def update_hparam_info(self, hparam_info):
    hparam_info.type = api_pb2.DATA_TYPE_FLOAT64  # TODO(#1998): Add int dtype.
    hparam_info.domain_interval.min_value = self._min_value
    hparam_info.domain_interval.max_value = self._max_value

  def sample_uniform(self, rng=random):
    return rng.randint(self._min_value, self._max_value)


class RealInterval(Domain):
  """A domain that takes on all real values in a closed interval."""

  def __init__(self, min_value=None, max_value=None):
    if not isinstance(min_value, float):
      raise ValueError("min_value must be an float: %r" % min_value)
    if not isinstance(max_value, float):
      raise ValueError("max_value must be an float: %r" % max_value)
    if min_value > max_value:
      raise ValueError("%r > %r" % (min_value, max_value))
    self._min_value = min_value
    self._max_value = max_value

  def __str__(self):
    return "[%s, %s]" % (self._min_value, self._max_value)

  def __repr__(self):
    return "RealInterval(%r, %r)" % (self._min_value, self._max_value)

  @property
  def dtype(self):
    return float

  def update_hparam_info(self, hparam_info):
    hparam_info.type = api_pb2.DATA_TYPE_FLOAT64
    hparam_info.domain_interval.min_value = self._min_value
    hparam_info.domain_interval.max_value = self._max_value

  def sample_uniform(self, rng=random):
    return rng.uniform(self._min_value, self._max_value)


class Discrete(Domain):
  """A domain that takes on a fixed set of values.

  These values may be of any (single) domain type.
  """

  def __init__(self, values, dtype=None):
    self._values = sorted(values)
    if dtype is None:
      if self._values:
        dtype = type(self._values[0])
      else:
        raise ValueError("Empty domain with no dtype specified")
    if dtype not in (int, float, bool, str):
      raise ValueError("Unknown dtype: %r" % dtype)
    self._dtype = dtype
    for value in self._values:
      if not isinstance(value, self._dtype):
        raise ValueError(
            "dtype mismatch: not isinstance(%r, %r)"
            % (value, self._dtype)
        )

  def __str__(self):
    return "{%s}" % (", ".join(repr(x) for x in self._values))

  def __repr__(self):
    return "Discrete(%r)" % (self._values,)

  @property
  def dtype(self):
    return self._dtype

  def update_hparam_info(self, hparam_info):
    hparam_info.type = {
        int: api_pb2.DATA_TYPE_FLOAT64,  # TODO(#1998): Add int dtype.
        float: api_pb2.DATA_TYPE_FLOAT64,
        bool: api_pb2.DATA_TYPE_BOOL,
        str: api_pb2.DATA_TYPE_STRING,
    }[self._dtype]
    hparam_info.ClearField("domain_discrete")
    hparam_info.domain_discrete.extend(self._values)

  def sample_uniform(self, rng=random):
    return rng.choice(self._values)


@six.add_metaclass(abc.ABCMeta)
class Metric(object):
  def __init__(self, proto):
    self._proto = proto

  def as_proto(self):
    result = api_pb2.MetricInfo()
    result.MergeFrom(self._proto)
    return result


class SummaryMetric(Metric):
  def __init__(
      self,
      tag,
      display_name=None,
      group=None,
      description=None,
      dataset_type=None,
  ):
    proto = api_pb2.MetricInfo(
        name=api_pb2.MetricName(group=group, tag=tag),
        display_name=display_name,
        description=description,
        dataset_type=dataset_type,
    )
    super(_KerasMetric, self).__init__(proto)


class _KerasMetric(Metric):
  def __init__(self, dataset_type, tag, display_name=None, keras_version=None):
    # TODO(wchargin): Figure out `keras_version` semantics? Want to be
    # stable here in the face of further changes to output directory
    # structure (like the `.` -> {`logs`, `train`} change).
    del keras_version

    group = {
        api_pb2.DATASET_TRAINING: "train",
        api_pb2.DATASET_VALIDATION: "validation",
    }[dataset_type]
    if display_name is None:
      display_name = tag
      for prefix in ("epoch_", "batch_"):
        if display_name.startswith(prefix):
          display_name = display_name[len(prefix):]
          break
      display_name = "%s (%s)" % (display_name, group)
    proto = api_pb2.MetricInfo(
        name=api_pb2.MetricName(group=group, tag=tag),
        display_name=display_name,
        dataset_type=dataset_type,
    )
    super(_KerasMetric, self).__init__(proto)


class KerasValidationMetric(_KerasMetric):
  def __init__(self, tag, display_name=None, keras_version=None):
    """TODO(@wchargin): Docs.

    Args:
      tag: Like `"epoch_accuracy"`.
    """
    super(KerasValidationMetric, self).__init__(
        dataset_type=api_pb2.DATASET_VALIDATION,
        tag=tag,
        display_name=display_name,
        keras_version=keras_version,
    )


class KerasTrainMetric(_KerasMetric):
  def __init__(self, tag, display_name=None, keras_version=None):
    """TODO(@wchargin): Docs.

    Args:
      tag: Like `"batch_accuracy"`.
    """
    super(KerasTrainMetric, self).__init__(
        dataset_type=api_pb2.DATASET_TRAINING,
        tag=tag,
        display_name=display_name,
        keras_version=keras_version,
    )


class Experiment(object):
  def __init__(
      self,
      hparams,
      metrics,
      user="",
      description="",
      time_created_secs=None,
  ):
    self._hparams = list(hparams)
    self._metrics = list(metrics)
    self._user = user
    self._description = description
    if time_created_secs is None:
      time_created_secs = time.time()
    self._time_created_secs = time_created_secs

  @property
  def hparams(self):
    return list(self._hparams)

  @property
  def metrics(self):
    return list(self._metrics)

  @property
  def user(self):
    return self._user

  @property
  def description(self):
    return self._description

  # TODO(@wchargin): Consider defining `define_flags` to attach things
  # to `absl.flags.FLAGS`?

  def summary(self):
    hparam_infos = []
    for hparam in self._hparams:
      info = api_pb2.HParamInfo(
          name=hparam.name,
          description=hparam.description,
          display_name=hparam.display_name,
      )
      hparam.domain.update_hparam_info(info)
      hparam_infos.append(info)
    metric_infos = [metric.as_proto() for metric in self._metrics]
    return summary.experiment_pb(
        hparam_infos=hparam_infos,
        metric_infos=metric_infos,
        user=self._user,
        description=self._description,
        time_created_secs=self._time_created_secs,
    )

  def write_to(self, logdir):
    writer = tf.compat.v2.summary.create_file_writer(logdir)
    try:
      _write_summary(writer, self.summary())
    finally:
      writer.close()


class KerasCallback(tf.keras.callbacks.Callback):
  """Callback for logging hyperparameters to TensorBoard."""

  def __init__(
      self,
      logdir,
      hparams,
      group_name=None,
  ):
    """Create a callback for logging hyperparameters to TensorBoard.

    Each callback object is good for one session only.

    Args:
      logdir: The log directory for this session.
      hparams: A `dict` mapping hyperparameters to the values used in
        this session. Keys should be the names of `HParam` objects used
        in the `Experiment`, or the `HParam` objects themselves. Values
        should be Python `bool`, `int`, `float`, or `string` values,
        depending on the type of the hyperparameter.
      group_name: The name of the session group containing this session,
        as a string or `None`. If `None` or empty, the group name is
        taken to be the session ID.

    Raises:
      ValueError: If two entries in `hparams` share the same
        hyperparameter name.
    """
    self._hparams = _normalize_hparams(hparams)
    self._group_name = group_name if group_name is not None else ""
    self._writer = tf.compat.v2.summary.create_file_writer(logdir)

  def __del__(self):
    # I was closing `self._writer` on `__del__`, but was also hitting
    # a "NotFoundError: ...SummaryWriterInterface does not exist" in the
    # depths of Keras. This seems to go away when I don't close the
    # writer... TODO(@wchargin): Investigate?
    #self._writer.close()
    pass

  def on_train_begin(self, logs=None):
    del logs  # unused
    pb = summary.session_start_pb(self._hparams, group_name=self._group_name)
    _write_summary(self._writer, pb)

  def on_train_end(self, logs=None):
    del logs  # unused
    pb = summary.session_end_pb(api_pb2.STATUS_SUCCESS)
    _write_summary(self._writer, pb)


def _normalize_hparams(hparams):
  """Normalize a dict of hparam values.

  Args:
    hparams: A `dict` whose keys are `HParam` objects and/or strings
      representing hyperparameter names, and whose values are
      hyperparameter values. No two keys may have the same name.

  Returns:
    A `dict` whose keys are hyperparameter names (as strings) and whose
    values are the corresponding hyperparameter values.
  """
  result = {}
  for (k, v) in six.iteritems(hparams):
    if isinstance(k, HParam):
      k = k.name
    if k in result:
      raise ValueError("multiple values specified for hparam %r" % k)
    result[k] = v
  return result


def _write_summary(writer, summary_pb):
  """TODO(@wchargin): Is this really the best way?"""
  with writer.as_default():
    event = tf.compat.v1.Event(summary=summary_pb).SerializeToString()
    tf.compat.v2.summary.import_event(event)
    writer.flush()
