#!/usr/bin/env bash
# Idempotent Cloud Agent install script for EasyDensity.jl.
# - Ensures a pinned Julia 1.10 toolchain is available.
# - Develops the sibling EasyHybrid.jl checkout (repository dependency) when present.
# - Instantiates and precompiles the project environment.
set -euo pipefail

JULIA_VERSION="1.10.12"
JULIA_PREFIX="/opt/julia"

# Resolve the repository root (directory that contains this .cursor/ folder).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 1. Ensure Julia is installed (idempotent).
if [ ! -x "${JULIA_PREFIX}/bin/julia" ]; then
  echo "Installing Julia ${JULIA_VERSION} ..."
  tmp_tar="$(mktemp --suffix=.tar.gz)"
  curl -fsSL \
    "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    -o "${tmp_tar}"
  sudo mkdir -p "${JULIA_PREFIX}"
  sudo tar -xzf "${tmp_tar}" -C "${JULIA_PREFIX}" --strip-components=1
  rm -f "${tmp_tar}"
fi
sudo ln -sf "${JULIA_PREFIX}/bin/julia" /usr/local/bin/julia
julia --version

# 2. Develop the local EasyHybrid.jl checkout when it is available alongside this repo.
#    Cloud Agents materialize repositoryDependencies next to the primary repo, so the
#    sibling path is the common case; fall back to the registered package otherwise.
HYBRID_PATH=""
for candidate in "${REPO_ROOT}/../EasyHybrid.jl" "${REPO_ROOT}/../EasyHybrid"; do
  if [ -f "${candidate}/Project.toml" ]; then
    HYBRID_PATH="$(cd "${candidate}" && pwd)"
    break
  fi
done

cd "${REPO_ROOT}"
if [ -n "${HYBRID_PATH}" ]; then
  echo "Developing local EasyHybrid at ${HYBRID_PATH}"
  HYBRID_PATH="${HYBRID_PATH}" julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["HYBRID_PATH"])'
else
  echo "Local EasyHybrid checkout not found; using the registered package."
fi

# 3. Instantiate and precompile the project (idempotent; safe to re-run).
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "EasyDensity.jl environment is ready."
