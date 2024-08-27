\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# GitHub Actions Continuous Integration {#github_actions_guide}

\tableofcontents

# Testing SpECTRE with GitHub Actions CI {#github_actions_ci}

SpECTRE uses
[GitHub Actions](https://github.com/features/actions) for
testing the code.  Multiple build jobs (described below) are launched
each time a pull request is submitted or updated.  GitHub Actions will also
launch these build jobs each time you push to a branch on your fork of SpECTRE
if you enable it. GitHub Actions is also used to deploy releases of the code.

For pull requests, you can view the GitHub Actions CI build by clicking on the
`Checks` tab. Near the bottom of the `Conversation` tab a summary of the CI
results are presented. You can view all of the GitHub Actions runs by clicking
on the `Actions` section.

## What is tested {#what-is-tested}

The GitHub Actions report lists the build jobs which will each have either a
green check mark if it passes, a red `X` if it has failed, or a yellow
dot with a circle if the build is in progress.  Clicking on a build job will
display the log of the build.

The following build jobs are launched:
* CHECK_COMMITS runs the script `tools/CheckCommits.sh` and fails the build if
  any casing of the words in the list below is the first word of the commit
  message.  This allows developers to flag their commits with these keywords to
  indicate that a pull request should not be merged in its current state.
  - fixup
  - wip (for work in progress)
  - fixme
  - deleteme
  - rebaseme
  - testing
  - rebase
* CHECK_FILES runs the script `tools/CheckFiles.sh` (which also runs the script
  `tools/FileTestDefs.sh`). The checks fail if any of the following are true:
  - Any file,
    * contains a line over 80 characters (We allow exceptions for certain file
      types and inherently long strings like URLs and include lines.
      See `tools/FileTestDefs.sh` for the full list of exceptions.)
    * is missing the license line
    * does not end with a newline
    * contains a tab character
    * contains white space at the end of a line
    * contains a carriage return character
  - A `c++` header file (i.e., `*.hpp` or `*.tpp`) is missing `#%pragma once`
  - A `c++` file (i.e., `*.hpp`, `*.tpp`, or `*.cpp`) file,
    * includes `<iostream>` (useless when running in parallel)
    * includes `<lrtslock.h>` (use `<converse.h>` instead)
    * includes `"Utilities/TmplDebugging.hpp"` (used only for debugging)
    * includes any non-header `*.cpp` file
    * contains a `namespace` ending in `_details` (use `_detail`)
    * contains a `struct TD` or `class TD` (used only for debugging)
    * contains `std::enable_if` (use `Requires` instead)
    * contains `Ls` (use `List` instead)
    * contains additional text after `/*!` (does not render correctly in
      Doxygen)
    * contains the string `return Py_None;` (bug prone, use `Py_RETURN_NONE`
      instead)
    * contains `.ckLocal()` or `.ckLocalBranch()` (use `Parallel::local` or
      `Parallel::local_branch` instead)
  - A `c++` test,
    * uses `TEST_CASE` (use `SPECTRE_TEST_CASE` instead)
    * uses `Approx` (use `approx` instead)
  - A `CMakeLists.txt` file in `src`, but not in an Executables or
    Python-binding directory,
    * does not list a `C++` file that is present in the directory
    * lists a `C++` file that is not present in the directory
  - A `c++` or `python` file contains a `TODO` (case-insensitive) comment
  In addition, the CHECK_FILES job tests Python formatting, the release
  workflow, and other tools in `tools/`.
* "Check Python formatting" runs the `black` and `isort` formatters over the
  source code.
* RUN_CLANG_TIDY runs clang-tidy on the source code. This is done for both
  `Release` and `Debug` builds.
* TEST_CHECK_FILES runs `tools/CheckFiles.sh --test` which tests the checks
  performed in the CHECK_FILES build.
* The other builds compile the code and run the tests for both
  `Release` and `Debug` builds, for the `gcc` and `clang` compilers
  using a Linux OS, and the `AppleClang` compiler for `OS X`.
* Verify the documentation builds successfully. Builds of `develop` deploy the
  documentation to GitHub pages.

## How to perform the checks locally {#perform-checks-locally}

Before pushing to GitHub and waiting for GitHub Actions to perform the checks it
is useful to perform at least the following tests locally:
- **Unit tests:** Perform a `make unit-tests` and then execute `ctest -L Unit`
  to run all unit tests. As for `make` you can append a `-jN` flag to `ctest` to
  run in parallel on `N` cores. To run only a subset of the tests you can use
  one of the other keywords that the tests are labeled with, such as `ctest -L
  datastructures`. To run only particular tests you can also execute `ctest -R
  TEST_NAME` instead, where `TEST_NAME` is a regular expression matching the
  test identifiers such as `Unit.DataStructures.Mesh`. Pass the flag
  `--output-on-failure` to get output from failed tests. Consult `ctest -h` for
  further options.

  To run the input file tests you must build the executables using
  `make test-executables`. You can then run `ctest -LE unit` to run everything
  except for the unit tests, or `ctest` to run all tests.
- **clang-tidy:** In a clang build directory, run `make clang-tidy
  FILE=SOURCE_FILE` where `SOURCE_FILE` is a relative or absolute path to a
  `.cpp` file. To perform this check for all source files that changed in your
  pull request, `make clang-tidy-hash HASH=UPSTREAM_HEAD` where `UPSTREAM_HEAD`
  is the hash of the commit that your pull request is based on, usually the
  `HEAD` of the `upstream/develop` branch.
- **Python formatting:** Run `black --check .` and `isort --check-only .` over
  the repository. You can install these tools with `pip3 install -r
  support/Python/dev_requirements.txt`
- **Documentation:** To render the documentation for the current state
  of the source tree the command `make doc` (or `make doc-check` to
  highlight warnings) can be used, placing its result in the `docs`
  directory in the build tree.  Once code has been made into a pull
  request to GitHub, the documentation can be rendered locally using
  the `tools/pr-docs` script.  To view the documentation, simply open the
  `index.html` file in the `html` subdirectory in a browser. Some functionality
  requires a web server (e.g. citation popovers), so just run a
  `python3 -m http.server` in the `html` directory to enable this.
- The `gcc Debug` build runs code coverage for each GitHub Actions build.

## Troubleshooting {#github-actions-troubleshooting}

* Occasionally, a build job will fail because of a problem with GitHub Actions
  (e.g. it times out).  On the `Checks` tab you can restart all or only the
  failed jobs. In the top right corner there's a `Re-run jobs` menu, which also
  has `Re-run failed jobs`. This button is `Cancel workflow` during the build
  process. Note that these buttons are only available if you have write access
  to the repository (core developer status).
* GitHub Actions caches some things between builds.  Occasionally this may
  cause a problem leading to strange build failures.  For example, inexplicable
  segfaults on seemingly random tests or `Illegal instruction` failures. We have
  to be fairly lax with our caching policies, and so the cache can become stale
  and outdated when a new container is pushed, among other difficult to
  understand situations. You can rebuild the ccache by going to `Actions`, then
  select the `Tests` workflow on the left, click the `Run workflow` drop-down
  menu, and enter `yes` in the input field below the ccache discussion.

  If clearing the ccache doesn't help, it could be that a Docker image layer is
  not being updated. GitHub doesn't (yet) have a way to clear the cache, so
  instead we clobber it to force GitHub to eject all old caches, both ccache and
  Docker images, along with anything else. To do this go to `Actions`, select
  the `Clobber Cache` workflow, then run it on develop. This will dump 9.9GB of
  random data into the cache. The amount is specified in the `ClobberCache.yaml`
  workflow file and needs to be updated if GitHub increases their cache
  size. The current cache size limit is 10GB per repository.

  Note that starting these workflows is only possible if you have write access
  to the repository (core developer status).

## Precompiled Headers and ccache {#precompiled-headers-ccache}

Getting ccache to work with precompiled headers on GitHub Actions is a little
challenging. The header to be precompiled is
`${SPECTRE_SOURCE_DIR}/tools/SpectrePch.hpp` and is symbolically linked to
`${SPECTRE_BUILD_DIR}/SpectrePch.hpp`. The configuration that seems to work is
specifying the environment variables:

\code{.sh}
CCACHE_COMPILERCHECK=content
CCACHE_EXTRAFILES="${SPECTRE_SOURCE_DIR}/tools/SpectrePch.hpp"
CCACHE_IGNOREHEADERS=\
  "${SPECTRE_BUILD_DIR}/SpectrePch.hpp:${SPECTRE_BUILD_DIR}/SpectrePch.hpp.gch"
\endcode

## Caching Dependencies on macOS Builds {#caching-mac-os}

On macOS builds we cache all of our dependencies, like LIBXSMM and
Charm++. These are cached in `$HOME/mac_cache`. Ultimately this saves about
10-12 minutes even when compared to using ccache to cache the object files from
building the dependencies. We also cache `$HOME/Library/Caches/Homebrew`, which
is where Homebrew keeps the downloaded formulas. By caching the Homebrew bottles
we are able to avoid brew formulas building from source because a tarball of the
package was not available at the time.
