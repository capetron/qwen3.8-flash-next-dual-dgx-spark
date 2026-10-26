#!/bin/bash
# Smoke test for the Qwen3.8-Flash-Next service: model listed, one chat answer, one structured tool call.
# usage: [MODEL=name] scripts/smoke-test.sh [BASE_URL]      (default http://127.0.0.1:8888/v1, model qwen3.8-flash-next)
set -euo pipefail
BASE=${1:-http://127.0.0.1:8888/v1}
MODEL=${MODEL:-qwen3.8-flash-next}   # the --served-model-name of your service

echo "1/3 models"
curl -fsS "$BASE/models" | python3 -c 'import json,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]; print("  ", ids); sys.exit(0 if sys.argv[1] in ids else 1)' "$MODEL"

echo "2/3 chat"
curl -fsS "$BASE/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; c=(m.get("content") or "").strip(); print("  ", c[:120]); sys.exit(0 if c else 1)'
{"model": "$MODEL", "max_tokens": 512,
 "messages": [{"role": "user", "content": "Reply with one short sentence: what is 17 times 3?"}]}
JSON

echo "3/3 tool call"
curl -fsS "$BASE/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; t=m.get("tool_calls") or []; print("  ", [x["function"]["name"] for x in t]); sys.exit(0 if t and t[0]["function"]["name"]=="get_weather" else 1)'
{"model": "$MODEL", "max_tokens": 1024, "tool_choice": "auto",
 "messages": [{"role": "user", "content": "What is the weather in Raleigh right now?"}],
 "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
JSON

echo "OK: models, chat and tool call all answered"
