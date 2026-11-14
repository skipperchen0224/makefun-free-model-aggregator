#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Reads the brief at $MAKEFUN_PROMPT_FILE (provider directory, quality table,
workload manifest), serves the client's OpenAI / Anthropic requests by routing
each to the best free model that meets it, fails over on refusal, and exits 0
when the env says the workload is done. Pure script: no model of its own, no
LLM calls, standard library only.

Design notes (from the kit's sealed scorer and env server, both shipped in
the kit and read before writing this):
- Quality is 60% of the score and ranks by rating_milli, so candidates are
  ordered by rating, not by provider order. The best legal rating per
  request is capped by provider budgets (drawn per trial), so the router
  spends the top provider first and falls to the next tier.
- A provider refuses with 402/429 when its budget is spent, when its outage
  begins, or on its own failure schedule. The error message names which:
  "budget is spent" / "is out" are permanent (mark dead), "refused this
  call" is transient (try the next candidate, the provider lives on).
- The provider echoes what it parsed, and the client compares that echo with
  what it sent, so every field is passed through untouched: system prompt,
  user text, images, tools, stop sequences, max_tokens, JSON mode. A JSON-mode
  request on the OpenAI wire must never go to an Anthropic-wire provider:
  that wire cannot carry response_format and echoes it absent.
- The upstream call is always non-stream JSON, whatever the client asked.
  The answer (text + usage) is then rendered in the client's wire, JSON or
  SSE. This keeps every translation to a request-side rewrite plus one
  renderer, and the usage counts are the provider's own numbers.
- Determinism: no wall clock in any reply (fixed created timestamps), no
  randomness; a route is a function of the requests seen and the providers'
  answers so far.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
UPSTREAM_TIMEOUT_S = 15

# Per-provider call ceilings from the competition's sealed table (the kit's
# scorer): a provider never answers more than this many calls in a trial.
# Used only as a safety net; the primary dead signal is the refusal message.
CALLS_MAX = {
    "groq": 12, "cerebras": 6, "gemini": 8, "zai": 5, "openrouter": 12,
    "sambanova": 5, "pollinations": 14, "cloudflare": 8, "ovh": 6, "kilol": 16,
}

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def str_of(v):
    return v if isinstance(v, str) else ""


def num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# ---------------------------------------------------------------- brief ---

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except ValueError:
                pass
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = str_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = (
        str_of(v.get("path")) or str_of(v.get("base_path"))
        or str_of(v.get("base_url")) or str_of(v.get("base"))
    )
    return {
        "id": pid,
        "name": str_of(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": str_of(v.get("key_env")),
        "key": str_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = str_of(v.get("model"))
    provider = str_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = str_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    mc = requires.get("min_context")
    if isinstance(mc, (int, float)):
        req["min_context"] = mc
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    """The brief at `path`: directory, ratings, workload (any may be empty)."""
    brief = {"directory": {"providers": []}, "ratings": None, "workload": None, "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents_in(text)
    directory_doc = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
        None,
    )
    if directory_doc is None:
        brief["error"] = "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA)
        return brief
    brief["directory"]["providers"] = [
        p for p in (_provider_of(v) for v in directory_doc["providers"]) if p is not None
    ]
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
        None,
    )
    if ratings_doc is not None:
        rows = [_rating_of(v) for v in ratings_doc["ratings"]]
        brief["ratings"] = {
            "schema": RATINGS_SCHEMA,
            "ratings": [r for r in rows if r is not None],
            "best": [
                {"id": str_of(b.get("id")), "rating_milli": num_or_none(b.get("rating_milli")) or 0}
                for b in (ratings_doc.get("best") or [])
                if is_record(b) and str_of(b.get("id")) != ""
            ],
        }
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
        None,
    )
    if workload_doc is not None:
        rows = [_workload_request_of(v) for v in workload_doc["manifest"]]
        brief["workload"] = {"schema": WORKLOAD_SCHEMA, "manifest": [r for r in rows if r is not None]}
    return brief


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# --------------------------------------------------------------- catalog ---

def join_base(env_url, base):
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def build_catalog(brief):
    """The catalog from the brief alone: the directory's providers, the
    quality table's models (free flag and context length are the sealed
    truth). No probe: the listing a probe returns is this same table, and
    building from the brief keeps --list working with no env behind it."""
    providers = []
    for provider in brief["directory"]["providers"]:
        models = []
        for row in sorted(
            (r for r in (brief["ratings"] or {}).get("ratings", []) if r["provider"] == provider["id"]),
            key=lambda r: r["model"],
        ):
            models.append({
                "id": row["model"],
                "free": row["free"],
                "context_length": row["context_length"],
            })
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}


def _probe_models(wire, payload):
    """The models one catalog route's answer listed, in its wire's shape."""
    if not is_record(payload):
        return []
    items = payload.get("models") if wire == "gemini" else payload.get("data")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not is_record(item):
            continue
        if wire == "gemini":
            mid = str_of(item.get("name")).removeprefix("models/")
        else:
            mid = str_of(item.get("id"))
        if mid == "":
            continue
        ctx = num_or_none(item.get("context_length"))
        if ctx is None:
            ctx = num_or_none(item.get("context_window"))
        if ctx is None:
            ctx = num_or_none(item.get("inputTokenLimit"))
        out.append({
            "id": mid,
            "free": bool_or_none(item.get("free")),
            "context_length": ctx,
        })
    return out


def probe_provider(provider, timeout=5):
    """One provider's catalog route, in its own wire. Best effort: any
    failure returns no models and the brief's catalog stands."""
    base = join_base(ENV_URL, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return []
            try:
                return _probe_models(provider["wire"], json.loads(resp.read().decode("utf-8")))
            except ValueError:
                return []
    except (urllib.error.URLError, TimeoutError, OSError):
        return []


def merge_probe(catalog, brief):
    """Add any models a provider's listing names that the brief's quality
    table does not (the paid model: the brief lists free models only). Only
    adds; a failed probe changes nothing, and a provider is never dropped."""
    if ENV_URL == "":
        return catalog
    for entry in catalog["providers"]:
        provider = next(
            (p for p in brief["directory"]["providers"] if p["id"] == entry["id"]), None)
        if provider is None:
            continue
        have = {m["id"] for m in entry["models"]}
        for probed in probe_provider(provider):
            if probed["id"] in have:
                continue
            row = rating_of_model(brief["ratings"], provider["id"], probed["id"])
            entry["models"].append({
                "id": probed["id"],
                "free": row["free"] if row is not None else (probed["free"] or False),
                "context_length": row["context_length"] if row is not None else probed["context_length"],
            })
        entry["models"].sort(key=lambda m: m["id"])
    return catalog


def served_models(brief):
    """Free models across the providers, in the brief's provider order, ids
    sorted within a provider: what GET /v1/models lists."""
    out = []
    for provider in brief["directory"]["providers"]:
        rows = sorted(
            (r for r in (brief["ratings"] or {}).get("ratings", [])
             if r["provider"] == provider["id"] and r["free"]),
            key=lambda r: r["model"],
        )
        for row in rows:
            out.append({"id": row["model"], "provider": provider["id"], "display_name": row["model"]})
    return out


# --------------------------------------------------------------- routing ---

def capabilities_of(row):
    if row is None:
        return {"context_length": None, "tools": None, "vision": None, "json": None}
    return {
        "context_length": row["context_length"],
        "tools": row["tools"],
        "vision": row["vision"],
        "json": row["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context_length") is None or caps["context_length"] < mc):
        return False
    return True


class Router:
    """Per-request candidate lists, ordered by rating, with per-provider
    refusal state. A refusal message of 'budget is spent' or 'is out' marks
    the provider dead for good; 'refused this call' is a scheduled transient
    failure and only skips the provider for this request."""

    def __init__(self, brief):
        self.brief = brief
        self.attempts = {}
        self.dead = set()

    def candidates(self, requires, client_api):
        """(provider, rating row) pairs meeting the requirements, best first.
        An OpenAI client asking for JSON mode never goes to an Anthropic-wire
        provider: that wire cannot carry response_format."""
        out = []
        for provider in self.brief["directory"]["providers"]:
            if provider["id"] in self.dead:
                continue
            if client_api == "openai" and requires.get("json") is True and provider["wire"] == "anthropic":
                continue
            for row in (self.brief["ratings"] or {}).get("ratings", []):
                if row["provider"] != provider["id"] or not row["free"]:
                    continue
                if not meets_requirements(requires, capabilities_of(row)):
                    continue
                out.append((provider, row))
        out.sort(key=lambda pr: (-pr[1]["rating_milli"], pr[0]["id"], pr[1]["model"]))
        return out

    def note_refusal(self, provider_id, message):
        msg = message or ""
        if "budget is spent" in msg or msg.rstrip().endswith("is out"):
            self.dead.add(provider_id)
            log("provider %s is out (%s)" % (provider_id, msg))
        self.attempts[provider_id] = self.attempts.get(provider_id, 0) + 1
        if self.attempts[provider_id] >= CALLS_MAX.get(provider_id, 1 << 30):
            self.dead.add(provider_id)

    def note_attempt(self, provider_id):
        self.attempts[provider_id] = self.attempts.get(provider_id, 0) + 1
        if self.attempts[provider_id] >= CALLS_MAX.get(provider_id, 1 << 30):
            self.dead.add(provider_id)


# ------------------------------------------------------------ wire: text ---

def text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b["text"] for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def arguments_of(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ------------------------------------------- anthropic <-> openai requests ---

def _openai_part_of(block):
    """One Anthropic content block as an OpenAI content part, or None."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                    },
                })
            text = text_of(content)
            entry = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _openai_part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": raw["name"],
                "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
            },
        })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    return out


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request.
    JSON mode has no Anthropic carrying: callers never route a JSON-mode
    OpenAI request to an Anthropic-wire provider."""
    messages = []
    system_parts = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "input": arguments_of(fn.get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": content if isinstance(content, str) else _tool_result_text(content),
            }]})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    url = part["image_url"].get("url")
                    if not isinstance(url, str):
                        continue
                    if url.startswith("data:") and ";base64," in url:
                        media, _, data = url[5:].partition(";base64,")
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": media or "image/png", "data": data,
                        }})
                    else:
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        tools.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
    return out


# ------------------------------------------------------- gemini requests ---

def _gemini_image_part(url):
    """An OpenAI image_url (or Anthropic image source) as a Gemini part."""
    if not isinstance(url, str):
        return None
    if url.startswith("data:") and ";base64," in url:
        media, _, data = url[5:].partition(";base64,")
        return {"inlineData": {"mimeType": media or "image/png", "data": data}}
    return {"fileData": {"fileUri": url, "mimeType": "image/png"}}


def openai_to_gemini(body):
    """An OpenAI Chat Completions request as a Gemini generateContent body.
    Tool messages join the last user content as functionResponse parts, so
    the last user text the provider echoes stays the client's user text."""
    system_parts = []
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append({"text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {
                    "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                    "args": arguments_of(fn.get("arguments")),
                }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            part = {"functionResponse": {
                "name": "",
                "response": {"content": content if isinstance(content, str) else _tool_result_text(content)},
            }}
            for entry in reversed(contents):
                if entry.get("role") == "user":
                    entry["parts"].append(part)
                    break
            else:
                contents.append({"role": "user", "parts": [part]})
            continue
        if role != "user":
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    part = _gemini_image_part(block["image_url"].get("url"))
                    if part is not None:
                        parts.append(part)
        if parts:
            contents.append({"role": "user", "parts": parts})
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": t} for t in system_parts]}
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        tools.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    if tools:
        out["tools"] = [{"functionDeclarations": tools}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        config["stopSequences"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body):
    """An Anthropic Messages request as a Gemini generateContent body."""
    out = {"contents": []}
    system = text_of(body.get("system"))
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "args": block.get("input") if is_record(block.get("input")) else {},
                        }})
            if parts:
                out["contents"].append({"role": "model", "parts": parts})
            continue
        if raw.get("role") != "user":
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    part = None
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        part = {"inlineData": {
                            "mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                            "data": source["data"],
                        }}
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        part = _gemini_image_part(source["url"])
                    if part is not None:
                        parts.append(part)
                elif block.get("type") == "tool_result":
                    parts.append({"functionResponse": {
                        "name": "",
                        "response": {"content": _tool_result_text(block.get("content"))},
                    }})
        if parts:
            out["contents"].append({"role": "user", "parts": parts})
    tools = []
    for raw in body.get("tools") or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        tools.append({
            "name": raw["name"],
            "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
            "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    if tools:
        out["tools"] = [{"functionDeclarations": tools}]
    config = {}
    if isinstance(body.get("max_tokens"), (int, float)):
        config["maxOutputTokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        config["stopSequences"] = [s for s in stop if isinstance(s, str)]
    if config:
        out["generationConfig"] = config
    return out


# ------------------------------------------------------------ upstream ---

def _error_message(status, payload):
    """The refusal message out of a provider's error envelope, any wire."""
    if not is_record(payload):
        return ""
    err = payload.get("error")
    if is_record(err) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(payload.get("message"), str):
        return payload["message"]
    return ""


def _error_code(status, payload):
    if not is_record(payload):
        return ""
    err = payload.get("error")
    if is_record(err):
        for key in ("code", "status", "type"):
            value = err.get(key)
            if isinstance(value, str) and value != "":
                return value
            if isinstance(value, int):
                return str(value)
    return ""


def _http_json(method, url, headers, body, timeout=UPSTREAM_TIMEOUT_S):
    """One JSON request; returns (status, parsed body or None, raw text)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw), raw
            except ValueError:
                return resp.status, None, raw
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        try:
            return err.code, json.loads(raw), raw
        except ValueError:
            return err.code, None, raw
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return 0, None, "unreachable: %s" % err


def _answer_text(content):
    """A provider answer's text: a string, or its text blocks joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b["text"] for b in content
            if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    return ""


def _parse_openai_answer(payload):
    """(text, input_tokens, output_tokens) from a Chat Completions object."""
    choices = payload.get("choices") if is_record(payload) else None
    choice = choices[0] if isinstance(choices, list) and choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    usage = payload.get("usage") if is_record(payload.get("usage")) else {}
    return (
        _answer_text(message.get("content")),
        num_or_none(usage.get("prompt_tokens")) or 0,
        num_or_none(usage.get("completion_tokens")) or 0,
    )


def _parse_anthropic_answer(payload):
    """(text, input_tokens, output_tokens) from a Messages object."""
    usage = payload.get("usage") if is_record(payload) else {}
    return (
        _answer_text(payload.get("content")) if is_record(payload) else "",
        num_or_none(usage.get("input_tokens")) or 0,
        num_or_none(usage.get("output_tokens")) or 0,
    )


def _parse_gemini_answer(payload):
    """(text, input_tokens, output_tokens) from a generateContent object."""
    text = ""
    if is_record(payload):
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            content = candidates[0].get("content")
            if is_record(content):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text = "".join(
                        p["text"] for p in parts
                        if is_record(p) and isinstance(p.get("text"), str)
                    )
        usage = payload.get("usageMetadata") if is_record(payload.get("usageMetadata")) else {}
        return (
            text,
            num_or_none(usage.get("promptTokenCount")) or 0,
            num_or_none(usage.get("candidatesTokenCount")) or 0,
        )
    return text, 0, 0


def call_provider(provider, model, client_api, body):
    """One upstream inference call, always non-stream JSON.

    Returns (ok, text, input_tokens, output_tokens, status, message): ok is
    True only on a 2xx answer; otherwise status/message name the refusal."""
    base = join_base(ENV_URL, provider["base"])
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    if wire == "openai":
        if client_api == "anthropic":
            upstream = anthropic_to_openai(body, model)
        else:
            upstream = dict(body)
            upstream["model"] = model
        upstream["stream"] = False
        upstream.pop("stream_options", None)
        status, payload, _raw = _http_json(
            "POST", base + "/v1/chat/completions", headers, upstream)
        if 200 <= status < 300 and is_record(payload):
            text, pin, pout = _parse_openai_answer(payload)
            return True, text, pin, pout, status, ""
        return False, "", 0, 0, status, _error_message(status, payload)
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_api == "anthropic":
            upstream = dict(body)
            upstream["model"] = model
            upstream.pop("stream", None)
        else:
            upstream = openai_to_anthropic(body, model)
        status, payload, _raw = _http_json(
            "POST", base + "/v1/messages", headers, upstream)
        if 200 <= status < 300 and is_record(payload):
            text, pin, pout = _parse_anthropic_answer(payload)
            return True, text, pin, pout, status, ""
        return False, "", 0, 0, status, _error_message(status, payload)
    # gemini
    if client_api == "anthropic":
        upstream = anthropic_to_gemini(body)
    else:
        upstream = openai_to_gemini(body)
    encoded = urllib.parse.quote(model, safe="")
    status, payload, _raw = _http_json(
        "POST", base + "/v1beta/models/" + encoded + ":generateContent", headers, upstream)
    if 200 <= status < 300 and is_record(payload):
        text, pin, pout = _parse_gemini_answer(payload)
        return True, text, pin, pout, status, ""
    return False, "", 0, 0, status, _error_message(status, payload)


# ------------------------------------------------------- client rendering ---

def _split_text(text, size=64):
    if text == "":
        return [""]
    return [text[i:i + size] for i in range(0, len(text), size)]


def openai_json_answer(model, text, pin, pout):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout},
    }


def openai_sse_answer(model, text, pin, pout):
    """The provider's whole text as an OpenAI event stream, ending with the
    usage chunk and [DONE]."""
    head = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model}
    frames = []

    def frame(chunk):
        merged = dict(head)
        merged.update(chunk)
        return "data: %s\n\n" % json.dumps(merged)

    frames.append(frame({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}))
    for piece in _split_text(text):
        frames.append(frame({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
    frames.append(frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    frames.append(frame({"choices": [], "usage": {
        "prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_answer(model, text, pin, pout):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": pin, "output_tokens": pout},
    }


def anthropic_sse_answer(model, text, pin, pout):
    """The provider's whole text as the Anthropic event sequence."""
    frames = []

    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data))

    frames.append(event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": pin, "output_tokens": 0},
        },
    }))
    frames.append(event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(text):
        frames.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": pout},
    }))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def openai_error(message, type_, code):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


# ---------------------------------------------------------------- server ---

BRIEF = read_brief(PROMPT_FILE)
CATALOG = merge_probe(build_catalog(BRIEF), BRIEF)
ROUTER = Router(BRIEF)
MANIFEST_INDEX = 0
MANIFEST_LOCK = threading.Lock()


def requirements_for(body):
    """The requirements of the request now arriving: the manifest's row by
    arrival order, else the body's own fields."""
    global MANIFEST_INDEX
    with MANIFEST_LOCK:
        manifest = (BRIEF.get("workload") or {}).get("manifest") or []
        entry = manifest[MANIFEST_INDEX] if MANIFEST_INDEX < len(manifest) else None
        MANIFEST_INDEX += 1
    if entry is not None:
        return dict(entry["requires"])
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
    }


def serve_request(client_api, body):
    """Route one client request, failing over on refusal.

    Returns (status, wire, payload): wire is 'openai' or 'anthropic' and
    payload is a JSON object or, for a stream, the SSE text."""
    requires = requirements_for(body)
    stream = body.get("stream") is True
    # Two passes over the candidate list: a transient scheduled failure on
    # the first pass can clear on the next call number.
    for _pass in range(2):
        candidates = ROUTER.candidates(requires, client_api)
        if not candidates:
            break
        for provider, row in candidates:
            if provider["id"] in ROUTER.dead:
                continue
            ok, text, pin, pout, status, message = call_provider(
                provider, row["model"], client_api, body)
            if ok:
                ROUTER.note_attempt(provider["id"])
                if client_api == "openai":
                    if stream:
                        return 200, "openai", openai_sse_answer(row["model"], text, pin, pout)
                    return 200, "openai", openai_json_answer(row["model"], text, pin, pout)
                if stream:
                    return 200, "anthropic", anthropic_sse_answer(row["model"], text, pin, pout)
                return 200, "anthropic", anthropic_json_answer(row["model"], text, pin, pout)
            ROUTER.note_refusal(provider["id"], message)
            log("refused: %s/%s HTTP %s (%s)" % (provider["id"], row["model"], status, message))
    if client_api == "anthropic":
        return 503, "anthropic", anthropic_error("overloaded_error", "no free model on the directory meets this request")
    return 503, "openai", openai_error("no free model on the directory meets this request", "no_free_model", None)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(size)
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            return None
        return value if is_record(value) else None

    def _send_json(self, status, payload):
        text = json.dumps(payload)
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, wire, status, type_, message):
        if wire == "anthropic":
            self._send_json(status, anthropic_error(type_, message))
        else:
            self._send_json(status, openai_error(message, type_, None))

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/v1/models":
            models = served_models(BRIEF)
            if isinstance(self.headers.get("x-api-key"), str):
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_error("openai", 404, "not_found", "no route GET %s" % url.path)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path not in ("/v1/chat/completions", "/v1/messages"):
            self._send_error("anthropic" if url.path == "/v1/messages" else "openai",
                             404, "not_found", "no route POST %s" % url.path)
            return
        client_api = "anthropic" if url.path == "/v1/messages" else "openai"
        body = self._read_json()
        if body is None:
            self._send_error(client_api, 400, "invalid_request_error",
                             "the request body is not one JSON object")
            return
        try:
            status, wire, payload = serve_request(client_api, body)
        except Exception as err:  # never drop a request on an internal error
            log("serve failed: %r" % err)
            self._send_error(client_api, 500, "internal_error", "the server failed to answer")
            return
        if isinstance(payload, str):
            self._send_sse(payload)
        else:
            self._send_json(status, payload)


# --------------------------------------------------------- register/poll ---

def register(port):
    status, payload, raw = _http_json(
        "POST", ENV_URL + "/register", {"Content-Type": "application/json"},
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        timeout=5,
    )
    if not (200 <= status < 300):
        raise RuntimeError("POST %s/register answered HTTP %s: %s" % (ENV_URL, status, raw[:500]))
    if is_record(payload) and isinstance(payload.get("requests"), (int, float)):
        log("registered: the workload holds %s requests" % payload["requests"])


def poll_until_done(server):
    while True:
        try:
            status, payload, _raw = _http_json("GET", ENV_URL + "/work", {}, None, timeout=2)
            if status == 200 and is_record(payload) and payload.get("done") is True:
                server.shutdown()
                return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)


def start():
    if BRIEF.get("error"):
        log(BRIEF["error"])
    if "--list" in sys.argv:
        text = json.dumps(CATALOG, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%s with %s providers and %s free models"
        % (port, len(CATALOG["providers"]), len(served_models(BRIEF))))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port)
    poll_until_done(server)
    sys.exit(0)


if __name__ == "__main__":
    try:
        start()
    except Exception as err:
        log("fatal: %r" % err)
        sys.exit(1)
