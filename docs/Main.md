\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
\mainpage Introduction

\tableofcontents

\htmlonly
<p>
<a
href="https://github.com/sxs-collaboration/spectre/blob/develop/LICENSE.txt"><img
src="https://img.shields.io/badge/license-MIT-blue.svg"
alt="license"
data-canonical-src="https://img.shields.io/badge/license-MIT-blue.svg"
style="max-width:100%;"></a>

<a href="https://en.wikipedia.org/wiki/C%2B%2B#Standardization"
rel="nofollow"><img
src="https://img.shields.io/badge/c%2B%2B-17-blue.svg"
alt="Standard"
data-canonical-src="https://img.shields.io/badge/c%2B%2B-17-blue.svg"
style="max-width:100%;"></a>

<a href="https://github.com/sxs-collaboration/spectre/actions" rel="nofollow"><img
src="https://github.com/sxs-collaboration/spectre/workflows/Tests/badge.svg?branch=develop"
alt="Build Status"
data-canonical-src="https://github.com/sxs-collaboration/spectre/workflows/Tests/badge.svg?branch=develop"
style="max-width:100%;"></a>

<a href="https://codecov.io/gh/sxs-collaboration/spectre" rel="nofollow"><img
src="https://codecov.io/gh/sxs-collaboration/spectre/graph/badge.svg?token=yyJ3uBPUE2"
alt="codecov"
data-canonical-src="https://codecov.io/gh/sxs-collaboration/spectre/graph/badge.svg?token=yyJ3uBPUE2"
style="max-width:100%;"></a>

</p>

<img src="banner.png"/>
\endhtmlonly

# What is SpECTRE? {#intro_sec}

SpECTRE is an open-source code for multi-scale, multi-physics problems
in astrophysics and gravitational physics. In the future, we hope that
it can be applied to problems across discipline boundaries in fluid
dynamics, geoscience, plasma physics, nuclear physics, and
engineering. It runs at petascale and is designed for future exascale
computers.

SpECTRE is being developed in support of our collaborative Simulating
eXtreme Spacetimes (SXS) research program into the multi-messenger
astrophysics of neutron star mergers, core-collapse supernovae, and
gamma-ray bursts.

## Citing SpECTRE

Please cite SpECTRE in any publications that make use of its code or data. Cite
the latest version that you use in your publication. The DOI for this version
is:

- DOI: [\spectredoi](https://doi.org/\spectredoi)

You can cite this BibTeX entry in your publication:

\include citation.bib

To aid reproducibility of your scientific results with SpECTRE, we recommend you
keep track of the version(s) you used and report this information in your
publication. We also recommend you supply the YAML input files and, if
appropriate, any additional C++ code you wrote to compile SpECTRE executables as
supplemental material to the publication.

See our [publication policy](https://spectre-code.org/publication_policies.html)
for more information.

## Navigating the Documentation {#navigate_documentation_sec}

The SpECTRE documentation is organized into tutorials, developer guides, groups
of related code, namespaces, and files for easier navigation. These can all be
accessed by links in the menu bar at the top.

- For instructions on **installing SpECTRE** on personal computers and clusters
  consult the \ref installation "Installation" and \ref installation_on_clusters
  "Installation on clusters" pages, respectively. See the
  \ref spectre_build_system "build system" page for how to configure SpECTRE.
- If you are looking to **run simulations with SpECTRE** we recommend starting
  with the \ref tutorials "User Tutorials". The tutorials are designed to get
  users up and running with a simulation, as well as analyzing and visualizing
  the output.
- For people looking to **contribute to SpECTRE** there are tutorials on the
  \ref dev_guide "Dev Guide" page. For instance, the dev guide details the \ref
  code_review_guide "code review guidelines", how to \ref writing_unit_tests
  "write unit tests", how to \ref writing_good_dox "write documentation", and
  also provides information about C++ and the philosophy behind SpECTRE
  development.
- The [Code modules](modules.html) section contains groups of related code
  (managed through doxygen groups). For example, there is a group for all the
  data structures we use, a group for utility functions and classes, as well as
  groups for coordinate maps, domain creation, and many others. The
  [Code modules](modules.html) are designed to help developers discover existing
  functionality so that things are not re-implemented several times.
- You can also get an overview of the code base by namespace by visiting the
  [Namespaces](namespaces.html) section.
- To browse all files in the repository you can visit the
  [GitHub repository](https://github.com/sxs-collaboration/spectre) directly.

## Logos and visuals {#logos_and_visuals}

You can find logos and other visuals (e.g. to put on slides) in various formats,
colors, and sizes here:

- [Logos and visuals](https://drive.google.com/drive/folders/1GEvVrXNqlGojr3NBf7RMFo6BzFmf9Ww7?usp=share_link)
