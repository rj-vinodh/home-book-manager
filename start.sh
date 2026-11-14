#!/bin/bash
# All app execution must go through Docker Compose so the app runs in the supported
# container environment with the correct ports, volumes, and host networking.
set -e

echo "Starting the project via Docker Compose..."
docker compose up --build

# docker compose exits with the same status as the stack; if it returns non-zero,
# the shell returns non-zero and the user can see the container logs.
exit $?