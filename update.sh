#!/bin/sh
# Moves downloaded updates into the repository layout. Run from the project root: sh update.sh
# Accepts files.zip (Claude's "Download all" with several files) and single *.py / *.java files
# saved in the project root (a single file is downloaded without a zip).
set -e
if [ ! -f pyproject.toml ]; then
    echo "run from the project root"
    exit 1
fi
tmp=$(mktemp -d)
if [ -f files.zip ]; then
    python3 -m zipfile -e files.zip "$tmp"
    rm files.zip
fi
for f in *.py *.java; do
    if [ -f "$f" ]; then
        mv "$f" "$tmp/"
    fi
done
if [ -z "$(find "$tmp" -type f)" ]; then
    echo "nothing to update: save files.zip or the downloaded file in the project root"
fi
find "$tmp" -type f | while read -r f; do
    name=$(basename "$f")
    case "$name" in
        *.java) dest=src/gunlab/java/gunlab ;;
        train_mlp.py) dest=examples ;;
        *.py) dest=src/gunlab ;;
        README.md | pyproject.toml | gunlab.example.toml | .gitignore | update.sh) dest=. ;;
        *) echo "skipped $name"; continue ;;
    esac
    mkdir -p "$dest"
    # copy, then rename: replaces the file atomically, also when it is this running script
    cp "$f" "$dest/$name.new"
    mv "$dest/$name.new" "$dest/$name"
    echo "updated $dest/$name"
done
rm -rf "$tmp"
