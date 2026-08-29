#!/bin/sh
set -eu
GRADLE_VERSION=7.5.1
CACHE_DIR="${HOME}/.gradle/wrapper/dists/equation-solver-${GRADLE_VERSION}"
ZIP="$CACHE_DIR/gradle-${GRADLE_VERSION}-bin.zip"
DIST="$CACHE_DIR/gradle-${GRADLE_VERSION}"
if [ ! -x "$DIST/bin/gradle" ]; then
  mkdir -p "$CACHE_DIR"
  if [ ! -f "$ZIP" ]; then
    curl -fsSL "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -o "$ZIP"
  fi
  rm -rf "$DIST.tmp"
  mkdir -p "$DIST.tmp"
  unzip -q "$ZIP" -d "$DIST.tmp"
  mv "$DIST.tmp/gradle-${GRADLE_VERSION}" "$DIST"
  rmdir "$DIST.tmp"
fi
exec "$DIST/bin/gradle" "$@"
