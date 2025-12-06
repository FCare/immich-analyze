#!/bin/bash
set -e

# Validate required environment variables
required_vars=(
    "DB_USERNAME"
    "DB_PASSWORD"
    "DB_DATABASE_NAME"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Build safe arguments array
args=(
    "--combined"
    "--immich-root" "/data"
    "--postgres-url" "postgresql://$DB_USERNAME:$DB_PASSWORD@database:5432/$DB_DATABASE_NAME"
)

# Add optional configuration safely
if [ -n "$IMMICH_ANALYZE_OLLAMA_HOSTS" ]; then
    ARGS="$ARGS --ollama-hosts $IMMICH_ANALYZE_OLLAMA_HOSTS"
fi

if [ -n "$IMMICH_ANALYZE_MODEL_NAME" ]; then
    args+=("--model-name" "$IMMICH_ANALYZE_MODEL_NAME")
fi

if [ "${IMMICH_ANALYZE_IGNORE_EXISTING:-false}" = "true" ]; then
    args+=("--ignore-existing")
fi

# Numeric validations
if [[ "$IMMICH_ANALYZE_MAX_CONCURRENT" =~ ^[0-9]+$ ]]; then
    args+=("--max-concurrent" "$IMMICH_ANALYZE_MAX_CONCURRENT")
fi

if [[ "$IMMICH_ANALYZE_TIMEOUT" =~ ^[0-9]+$ ]]; then
    args+=("--timeout" "$IMMICH_ANALYZE_TIMEOUT")
fi

# Execute with proper signal handling
exec immich-analyze "${args[@]}"
