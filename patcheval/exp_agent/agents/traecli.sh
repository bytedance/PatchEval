# TraeCLI / TraeX agent adapter. Source this file from run_infer.sh.
#
# Required:
#   TRAE_BIN=/path/to/traex
#   TRAE_CONFIG=/path/to/traecli-runtime/trae-home/<profile>.traecli.toml
#
# TRAE_CONFIG is the profile/config input. The adapter derives:
#   TRAE_HOME_SRC = dirname(TRAE_CONFIG)
#   TRAE_PROFILE  = basename(TRAE_CONFIG) without .traecli.toml

: "${TRAE_BIN:?Set TRAE_BIN to the TraeX executable, e.g. /path/to/bin/traex}"
: "${TRAE_CONFIG:?Set TRAE_CONFIG to a TraeCLI profile config file, e.g. /path/to/traecli-runtime/trae-home/doubao-8hp8j.traecli.toml}"

TRAE_BIN="$(realpath "$TRAE_BIN")"
if [[ ! -x "$TRAE_BIN" ]]; then
  echo "TRAE_BIN does not exist or is not executable: $TRAE_BIN" >&2
  exit 1
fi
TRAE_CONFIG="$(realpath "$TRAE_CONFIG")"
if [[ ! -f "$TRAE_CONFIG" ]]; then
  echo "TRAE_CONFIG does not exist or is not a file: $TRAE_CONFIG" >&2
  exit 1
fi
TRAE_HOME_SRC="$(cd "$(dirname "$TRAE_CONFIG")" && pwd)"
config_name="$(basename "$TRAE_CONFIG")"
if [[ "$config_name" == *.traecli.toml ]]; then
  TRAE_PROFILE="${config_name%.traecli.toml}"
else
  echo "TRAE_CONFIG must be named <profile>.traecli.toml: $TRAE_CONFIG" >&2
  exit 1
fi

AGENT_MOUNTS=(
  "${TRAE_BIN}:/usr/local/bin/traex:ro"
  "${TRAE_HOME_SRC}:/opt/trae-home-src:ro"
)
AGENT_EXTRA_ARGS=()
AGENT_COMMAND="tmp_trae_home=\"/tmp/trae-home-\${{PATCHAGENT_SESSION_ID:-traex}}\"; rm -rf \"\$tmp_trae_home\"; mkdir -p \"\$tmp_trae_home\"; cp -a /opt/trae-home-src/. \"\$tmp_trae_home\"/; RUST_LOG=off TRAE_HOME=\"\$tmp_trae_home\" /usr/local/bin/traex exec --profile ${TRAE_PROFILE} --json --skip-git-repo-check --ephemeral --ignore-rules -y --output-last-message /results/traex_last_message.txt - < {prompt_file}"
