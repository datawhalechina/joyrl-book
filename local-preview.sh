#!/usr/bin/env sh

set -eu

PORT="${1:-3002}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

cd "$SCRIPT_DIR"

echo "==> Building site..."
npm run build

echo "==> Serving build output on http://localhost:${PORT}"
exec ./node_modules/.bin/docusaurus serve --dir build --port "${PORT}"
