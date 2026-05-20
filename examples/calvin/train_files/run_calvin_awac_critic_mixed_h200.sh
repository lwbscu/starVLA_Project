#!/usr/bin/env bash
# Deprecated wrapper: use h200_awac_critic_mixed_oneclick.sh (merge + allowlist + train).
exec bash "$(dirname "$0")/h200_awac_critic_mixed_oneclick.sh" "$@"
