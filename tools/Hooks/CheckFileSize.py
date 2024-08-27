#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import subprocess
import sys

# The maximum file-size for a file to be committed in kB
max_file_size = 200.0

# Files allowed to exceed the file-size limit
allowed_large_files = [
    "docs/config/cppreference-doxygen-web.tag.xml",
]

# The path to the git-binary:
git_executable = "@GIT_EXECUTABLE@"


def sizeof_fmt(num):
    """
    This function will return a human-readable filesize-string
    like "3.5 MB" for its given 'num'-parameter.
    From http://stackoverflow.com/questions/1094841
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


# Check all files in the staging-area:
text = subprocess.check_output(
    [git_executable, "status", "--porcelain", "-uno"], stderr=subprocess.STDOUT
).decode("utf-8")
file_list = text.splitlines()

# Check all files:
for file_s in file_list:
    if not os.path.isfile(file_s[3:]):
        continue
    elif file_s[3:] in allowed_large_files:
        continue
    stat = os.stat(file_s[3:])
    if stat.st_size > (max_file_size * 1024):
        print(
            "File '"
            + file_s[3:]
            + "' is too large to be committed. The file "
            "is %s and the limit is %s"
            % (sizeof_fmt(stat.st_size), sizeof_fmt(max_file_size * 1024))
        )
        sys.exit(1)

sys.exit(0)
