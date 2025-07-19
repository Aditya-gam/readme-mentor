#!/bin/bash

# Tag Release Script for readme-mentor
# This script creates and pushes a new version tag

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)

if [ -z "$CURRENT_VERSION" ]; then
    print_error "Could not determine version from pyproject.toml"
    exit 1
fi

print_status "Current version: $CURRENT_VERSION"

# Check if tag already exists
if git tag -l "v$CURRENT_VERSION" | grep -q "v$CURRENT_VERSION"; then
    print_warning "Tag v$CURRENT_VERSION already exists"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deleting existing tag v$CURRENT_VERSION"
        git tag -d "v$CURRENT_VERSION" || true
        git push origin ":refs/tags/v$CURRENT_VERSION" || true
    else
        print_status "Skipping tag creation"
        exit 0
    fi
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "There are uncommitted changes"
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Aborting tag creation"
        exit 0
    fi
fi

# Create the tag
print_status "Creating tag v$CURRENT_VERSION"
git tag -a "v$CURRENT_VERSION" -m "Release v$CURRENT_VERSION - CI Bootstrap"

# Push the tag
print_status "Pushing tag v$CURRENT_VERSION to remote"
git push origin "v$CURRENT_VERSION"

print_status "Successfully created and pushed tag v$CURRENT_VERSION"
print_status "You can now create a release on GitHub using this tag"
