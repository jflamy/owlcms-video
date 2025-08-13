#!/bin/bash
# Script to copy IOC 3-letter code flags for all countries in the Americas to competitions/asu/local/flags

set -e

SRC_DIR="flags/3letter_ioc"
DEST_DIR="competitions/asu/local/flags"

# List of IOC 3-letter codes for North, Central, and South American countries
CODES=(
  ARG BAH BAR BER BOL BRA CAN CHI COL CRC CUB DOM ECU ESA GUA GUY HAI HON JAM MEX NCA PAN PAR PER PUR SUR TTO URU USA VEN
)

mkdir -p "$DEST_DIR"

for CODE in "${CODES[@]}"; do
  # Copy all files matching the code (e.g., ARG.svg, ARG.png, etc.)
  for FILE in "$SRC_DIR/$CODE".*; do
    if [ -e "$FILE" ]; then
      cp -f "$FILE" "$DEST_DIR/"
    fi
  done
done

echo "Flags copied for: ${CODES[*]}"
