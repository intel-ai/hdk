#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import os

# We set these dlopen flags to allow calls from JIT code
# to HDK shared objects. Otherwise, such relocations would
# be unresolved and we would have calls by zero address.
# TODO: Is there a way to avoid this in the Python code?
if sys.platform == "linux":
    prev = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
elif sys.platform == "win32":
    if "JAVA_HOME" not in os.environ:
        raise RuntimeError(
            "HDK engine requires JAVA_HOME environment variable to point to Java runtime location."
        )
    os.add_dll_directory(os.path.join(os.environ["JAVA_HOME"], "bin", "server"))

from pyhdk._common import TypeInfo, buildConfig, initLogger
from pyhdk._execute import Executor
import pyhdk.sql as sql
import pyhdk.storage as storage
from pyhdk.hdk import init
from pyhdk._builder import QueryBuilder
from pyhdk.version import Version

if sys.platform == "linux":
    sys.setdlopenflags(prev)

__version__ = Version.str()
