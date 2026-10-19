# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-19

### Added

- First PyPI-ready project layout with a `src/` Python package and a `csrc/`
  C library.
- `scikit-build-core` + CMake build that compiles `libisodendron` and installs
  it next to the Python package.
- Platform wheels with the `py3-none-<platform>` tag.
- C tests through CTest and Python tests through pytest.
- GitHub Actions CI and cibuildwheel workflows.
- Examples and sample data under `examples/`.

### Changed

- The Python library is located with `importlib.resources` instead of
  `__file__` and source-tree heuristics.
- The package version now comes from `pyproject.toml` and is exposed by the C
  library through a CMake definition.
- Documentation updated for installation from PyPI and building from source.
