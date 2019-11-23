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
"""Tests for tensorboard.uploader.util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest


try:
    # python version >= 3.3
    from unittest import mock  # pylint: disable=g-import-not-at-top
except ImportError:
    import mock  # pylint: disable=g-import-not-at-top,unused-import


from google.protobuf import timestamp_pb2
from tensorboard.uploader import test_util
from tensorboard.uploader import util
from tensorboard import test as tb_test


class RateLimiterTest(tb_test.TestCase):
    def test_rate_limiting(self):
        rate_limiter = util.RateLimiter(10)
        fake_time = test_util.FakeTime(current=1000)
        with mock.patch.object(rate_limiter, "_time", fake_time):
            self.assertEqual(1000, fake_time.time())
            # No sleeping for initial tick.
            rate_limiter.tick()
            self.assertEqual(1000, fake_time.time())
            # Second tick requires a full sleep.
            rate_limiter.tick()
            self.assertEqual(1010, fake_time.time())
            # Third tick requires a sleep just to make up the remaining second.
            fake_time.sleep(9)
            self.assertEqual(1019, fake_time.time())
            rate_limiter.tick()
            self.assertEqual(1020, fake_time.time())
            # Fourth tick requires no sleep since we have no remaining seconds.
            fake_time.sleep(11)
            self.assertEqual(1031, fake_time.time())
            rate_limiter.tick()
            self.assertEqual(1031, fake_time.time())


class GetUserConfigDirectoryTest(tb_test.TestCase):
    def test_windows(self):
        with mock.patch.object(os, "name", "nt"):
            with mock.patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": "C:\\Users\\Alice\\AppData\\Local",
                    "APPDATA": "C:\\Users\\Alice\\AppData\\Roaming",
                },
            ):
                self.assertEqual(
                    "C:\\Users\\Alice\\AppData\\Local",
                    util.get_user_config_directory(),
                )
            with mock.patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": "",
                    "APPDATA": "C:\\Users\\Alice\\AppData\\Roaming",
                },
            ):
                self.assertEqual(
                    "C:\\Users\\Alice\\AppData\\Roaming",
                    util.get_user_config_directory(),
                )
            with mock.patch.dict(
                os.environ, {"LOCALAPPDATA": "", "APPDATA": "",}
            ):
                self.assertIsNone(util.get_user_config_directory())

    def test_non_windows(self):
        with mock.patch.dict(os.environ, {"HOME": "/home/alice"}):
            self.assertEqual(
                "/home/alice%s.config" % os.sep,
                util.get_user_config_directory(),
            )
            with mock.patch.dict(
                os.environ, {"XDG_CONFIG_HOME": "/home/alice/configz"}
            ):
                self.assertEqual(
                    "/home/alice/configz", util.get_user_config_directory()
                )


skip_if_windows = unittest.skipIf(os.name == "nt", "Unsupported on Windows")


class MakeFileWithDirectoriesTest(tb_test.TestCase):
    def test_windows_private(self):
        with mock.patch.object(os, "name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "Windows"):
                util.make_file_with_directories("/tmp/foo", private=True)

    def test_existing_file(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        os.makedirs(os.path.dirname(path))
        with open(path, mode="w") as f:
            f.write("foobar")
        util.make_file_with_directories(path)
        with open(path, mode="r") as f:
            self.assertEqual("foobar", f.read())

    def test_existing_dir(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        os.makedirs(os.path.dirname(path))
        util.make_file_with_directories(path)
        self.assertEqual(0, os.path.getsize(path))

    def test_nonexistent_leaf_dir(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        os.makedirs(os.path.dirname(os.path.dirname(path)))
        util.make_file_with_directories(path)
        self.assertEqual(0, os.path.getsize(path))

    def test_nonexistent_multiple_dirs(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        util.make_file_with_directories(path)
        self.assertEqual(0, os.path.getsize(path))

    def assertMode(self, mode, path):
        self.assertEqual(mode, os.stat(path).st_mode & 0o777)

    @skip_if_windows
    def test_private_existing_file(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        os.makedirs(os.path.dirname(path))
        with open(path, mode="w") as f:
            f.write("foobar")
        os.chmod(os.path.dirname(path), 0o777)
        os.chmod(path, 0o666)
        util.make_file_with_directories(path, private=True)
        self.assertMode(0o700, os.path.dirname(path))
        self.assertMode(0o600, path)
        with open(path, mode="r") as f:
            self.assertEqual("foobar", f.read())

    @skip_if_windows
    def test_private_existing_dir(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        os.makedirs(os.path.dirname(path))
        os.chmod(os.path.dirname(path), 0o777)
        util.make_file_with_directories(path, private=True)
        self.assertMode(0o700, os.path.dirname(path))
        self.assertMode(0o600, path)
        self.assertEqual(0, os.path.getsize(path))

    @skip_if_windows
    def test_private_nonexistent_leaf_dir(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        os.makedirs(os.path.dirname(os.path.dirname(path)))
        util.make_file_with_directories(path, private=True)
        self.assertMode(0o700, os.path.dirname(path))
        self.assertMode(0o600, path)
        self.assertEqual(0, os.path.getsize(path))

    @skip_if_windows
    def test_private_nonexistent_multiple_dirs(self):
        root = self.get_temp_dir()
        path = os.path.join(root, "foo", "bar", "qux.txt")
        util.make_file_with_directories(path, private=True)
        self.assertMode(0o700, os.path.dirname(path))
        self.assertMode(0o600, path)
        self.assertEqual(0, os.path.getsize(path))


class SetTimestampTest(tb_test.TestCase):
    def test_set_timestamp(self):
        pb = timestamp_pb2.Timestamp()
        t = 1234567890.007812500
        # Note that just multiplying by 1e9 would lose precision:
        self.assertEqual(int(t * 1e9) % int(1e9), 7812608)
        util.set_timestamp(pb, t)
        self.assertEqual(pb.seconds, 1234567890)
        self.assertEqual(pb.nanos, 7812500)


if __name__ == "__main__":
    tb_test.main()
