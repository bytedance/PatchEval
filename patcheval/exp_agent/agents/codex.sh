# Codex CLI agent adapter. Source this file from run_infer.sh.
#
# Required:
#   CODEX_BIN=/path/to/codex
#   CODEX_CONFIG=/path/to/codex-home/<profile>.config.toml
#
# CODEX_CONFIG is the single profile/config input. The adapter derives:
#   CODEX_HOME_SRC = dirname(CODEX_CONFIG)
#   CODEX_PROFILE  = basename(CODEX_CONFIG) without .config.toml

: "${CODEX_BIN:?Set CODEX_BIN to the Codex executable, e.g. /path/to/bin/codex}"
: "${CODEX_CONFIG:?Set CODEX_CONFIG to a Codex profile config file, e.g. /path/to/codex-home/gpt54-gggso.config.toml}"

CODEX_BIN="$(realpath "$CODEX_BIN")"
if [[ ! -x "$CODEX_BIN" ]]; then
  echo "CODEX_BIN does not exist or is not executable: $CODEX_BIN" >&2
  exit 1
fi

# The npm-installed `codex` command is a JavaScript launcher whose package
# dependencies are resolved relative to the installation tree. Mounting that
# launcher as a single file into a case container therefore does not work. On
# Linux, resolve it to the packaged native binary, which is self-contained and
# can safely be mounted at /usr/local/bin/codex.
if [[ "$(basename "$CODEX_BIN")" == "codex.js" ]]; then
  codex_package_root="$(cd "$(dirname "$CODEX_BIN")/.." && pwd)"
  native_candidates=(
    "${codex_package_root}"/node_modules/@openai/codex-linux-*/vendor/*/bin/codex
  )
  resolved_native=""
  for candidate in "${native_candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      if [[ -n "$resolved_native" ]]; then
        echo "Multiple native Codex binaries found under $codex_package_root; set CODEX_BIN to one explicitly" >&2
        exit 1
      fi
      resolved_native="$candidate"
    fi
  done
  if [[ -z "$resolved_native" ]]; then
    echo "CODEX_BIN points to the npm JavaScript launcher, but no Linux native binary was found under $codex_package_root" >&2
    echo "Set CODEX_BIN to the packaged Linux native binary explicitly" >&2
    exit 1
  fi
  CODEX_BIN="$(realpath "$resolved_native")"
fi
CODEX_CONFIG="$(realpath "$CODEX_CONFIG")"
if [[ ! -f "$CODEX_CONFIG" ]]; then
  echo "CODEX_CONFIG does not exist or is not a file: $CODEX_CONFIG" >&2
  exit 1
fi
CODEX_HOME_SRC="$(cd "$(dirname "$CODEX_CONFIG")" && pwd)"
config_name="$(basename "$CODEX_CONFIG")"
if [[ "$config_name" == *.config.toml ]]; then
  CODEX_PROFILE="${config_name%.config.toml}"
else
  echo "CODEX_CONFIG must be named <profile>.config.toml: $CODEX_CONFIG" >&2
  exit 1
fi

AGENT_MOUNTS=(
  "${CODEX_BIN}:/usr/local/bin/codex:ro"
  "${CODEX_HOME_SRC}:/opt/codex-home-src:ro"
)
AGENT_EXTRA_ARGS=()
AGENT_COMMAND="rm -rf /tmp/codex-home && cp -a /opt/codex-home-src /tmp/codex-home && CODEX_HOME=/tmp/codex-home /usr/local/bin/codex exec --profile ${CODEX_PROFILE} --json --dangerously-bypass-approvals-and-sandbox -C {workdir} < {prompt_file}"
