# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data provider that talks to a gRPC server."""

import contextlib

import grpc

from tensorboard import errors
from tensorboard.data import provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.data.proto import data_provider_pb2_grpc


class GrpcDataProvider(provider.DataProvider):
    """Data provider that talks over gRPC."""

    def __init__(self, addr, channel):
        self._addr = addr
        self._stub = data_provider_pb2_grpc.TensorBoardDataProviderStub(channel)

    def data_location(self, ctx, experiment_id):
        return "grpc://%s" % (self._addr,)

    def list_plugins(self, ctx, experiment_id):
        return ["scalars"]

    def list_runs(self, ctx, experiment_id):
        req = data_provider_pb2.ListRunsRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.ListRuns(req)
        return [
            provider.Run(
                run_id=run.id,
                run_name=run.name,
                start_time=_wall_time(run.start_time),
            )
            for run in res.runs
        ]

    def list_scalars(
        self, ctx, experiment_id, plugin_name, run_tag_filter=None
    ):
        req = data_provider_pb2.ListScalarsRequest()
        req.experiment_id = experiment_id
        req.plugin_filter.plugin_name = plugin_name
        if run_tag_filter is not None:
            if run_tag_filter.runs is not None:
                req.run_tag_filter.runs.runs[:] = sorted(run_tag_filter.runs)
            if run_tag_filter.tags is not None:
                req.run_tag_filter.tags.tags[:] = sorted(run_tag_filter.tags)
        with _translate_grpc_error():
            res = self._stub.ListScalars(req)
        return {
            run_entry.run_name: {
                tag_entry.tag_name: provider.ScalarTimeSeries(
                    max_step=tag_entry.time_series.max_step,
                    max_wall_time=_wall_time(
                        tag_entry.time_series.max_wall_time
                    ),
                    plugin_content=(
                        tag_entry.time_series.summary_metadata.plugin_data.content
                    ),
                    description=tag_entry.time_series.summary_metadata.summary_description,
                    display_name=tag_entry.time_series.summary_metadata.display_name,
                )
                for tag_entry in run_entry.tags
            }
            for run_entry in res.runs
        }

    def read_scalars(
        self,
        ctx,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        req = data_provider_pb2.ReadScalarsRequest()
        req.experiment_id = experiment_id
        req.plugin_filter.plugin_name = plugin_name
        if run_tag_filter is not None:
            if run_tag_filter.runs is not None:
                req.run_tag_filter.runs.runs[:] = sorted(run_tag_filter.runs)
            if run_tag_filter.tags is not None:
                req.run_tag_filter.tags.tags[:] = sorted(run_tag_filter.tags)
        req.downsample.num_points = downsample
        with _translate_grpc_error():
            res = self._stub.ReadScalars(req)

        result = {}
        for run_entry in res.runs:
            tags = {}
            result[run_entry.run_name] = tags
            for tag_entry in run_entry.tags:
                series = []
                tags[tag_entry.tag_name] = series
                d = tag_entry.data
                for (step, ts, value) in zip(d.step, d.wall_time, d.value):
                    pt = provider.ScalarDatum(
                        step=step, wall_time=_wall_time(ts), value=value,
                    )
                    series.append(pt)
        return result


@contextlib.contextmanager
def _translate_grpc_error():
    try:
        yield
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            raise errors.InvalidArgumentError(e.message())
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise errors.NotFoundError(e.message())
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            raise errors.PermissionDeniedError(e.message())
        raise


def _wall_time(ts):
    if ts is None:
        return None
    return ts.ToNanoseconds() / 1e9
