#!/bin/bash
# Full PoC run. Needs AWS_PROFILE and UAEF_POC_ACCOUNT_ID; replay.py aborts if the caller is a different account.
# UAEF_ENABLE_CACHING=false is a precaution: in the pinned commit the flag is only read in config.py.
set -eu
cd "$(dirname "$0")"
: "${AWS_PROFILE:?set AWS_PROFILE to the profile you want to run in}"
: "${UAEF_POC_ACCOUNT_ID:?set UAEF_POC_ACCOUNT_ID to its 12-digit account id}"
export AWS_REGION="${AWS_REGION:-us-east-1}" UAEF_ENABLE_CACHING=false
UAEF_LLM_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python replay.py --runs 3 --tag haiku-4-5
UAEF_LLM_MODEL_ID=us.anthropic.claude-opus-4-6-v1 uv run python replay.py --runs 1 --tag opus-4-6
