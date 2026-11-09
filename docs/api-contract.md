# lag serve API contract (version 1)

What `lag serve` promises its clients: the web console, `lag goals`, and the phone app.
[`contract/v1.json`](../contract/v1.json) is the same contract in a form tests can read,
and [`contract/examples/`](../contract/examples/) holds real streams recorded from a
running server. `tests/test_contract.py` keeps all three in step with the code.

## The rules

1. **Clients ignore what they don't know.** An unknown event type or an unknown field is
   skipped, never an error. That lets the server add events and fields without a new
   contract version.
2. **Nothing listed here is removed, renamed or retyped within a version.** That needs
   contract 2, and `GET /api/version` reports the version a server speaks. A client built
   for another version should say which side to update instead of failing oddly.
3. **Every failure carries an `error` string.** Check for it whatever the status:
   - refused requests answer 4xx;
   - `/api/run` and goal edits answer 200 with `error` (goal edits include the snapshot);
   - a stream's failure is its last event, type `error`.

## Connecting

The console port (default 8765) serves the Mac's own browser. The phone uses the phone
port (`lag serve --phone-port 8766`), reached through `tailscale serve`.

| On the phone port | |
|---|---|
| Key | Every request needs `Authorization: Bearer <device key>`, except `POST /api/pair` |
| 401 | No key, or a key that is not paired or was revoked |
| 403 | The key was paired by a different Tailscale user, or the request is cross-site |
| Chats | `/api/stream` and `/api/run` run in the sandbox |

## Endpoints

Bodies and responses are JSON. "Snapshot" means a `goals` event (below).

| Endpoint | Sends | Answers |
|---|---|---|
| `GET /api/version` | | `{lag, contract}` |
| `GET /api/state` | | `{version, ollama, chat_model, num_ctx, personas, tools, rag_tools, chroma, mcp}` |
| `POST /api/stream` | `{prompt, mode?, agents?, plan?, model?, route?, retrieval?, mcp?}` | A stream of events |
| `POST /api/run` | Same as `/api/stream` | `{answer, model}` when finished (blocking) |
| `GET /api/goals` | | Snapshot |
| `GET /api/goals/events` | | A stream: a snapshot on every change, plus the running goal's events, each with `goal` set to its id |
| `POST /api/goals` | `{text, done_when?, agents?}` (no agents: the orchestrator picks, recommended) | Snapshot |
| `DELETE /api/goals/{id}` | | Snapshot |
| `POST /api/goals/{id}/move?delta=N` | | Snapshot |
| `POST /api/goals/{id}/retry` | | Snapshot |
| `POST /api/goals/run`, `/api/goals/pause`, `/api/goals/skip`, `/api/goals/clear` | | Snapshot |
| `POST /api/goals/settings` | `{plan?, fail?, model?}` | Snapshot |
| `POST /api/pair` | `{code, name?}`, phone port only, no key | `{key, device}`. The key is in this reply and nowhere else. |
| `GET /api/devices` | | `{devices, you}`. `you` is this phone's name on the phone port, else null. |

Streams are server-sent events: each event is one line, `data: <json>`, followed by a
blank line. Lines starting with `:` are keepalives; skip them.

## Events

Each event is a JSON object with a `type`. The app draws the first group; agent mode
sends the second.

| Type | Fields | Meaning |
|---|---|---|
| `start` | `stages` [{name, model}], `loads` | A run begins |
| `stage` | `name`, `model`, `loaded`, `role`? | A stage starts talking; `loaded` means a cold model load |
| `thinking` | `text` | Part of the model's reasoning; show it collapsed |
| `token` | `text` | Part of the answer, cleaned for display. Draw tokens once per frame, not once per token. |
| `tool` | `name`, `args`, `pretty` | A tool call; `pretty` is how a person should read it |
| `tool_result` | `name`, `content` | What the tool returned (at most 2,000 characters) |
| `routing` | `lead`, `model`, `candidates` | The orchestrator is choosing agents |
| `routed` | `lead`, `agents` [{name, model}], `loads` | The orchestrator chose |
| `route_fallback` | `agents` | Its choice could not be read; the default roster runs |
| `trimmed` | `stage`, `model`, `window`, `dropped_chars` | Earlier work was cut to fit this stage |
| `no_tools` | `model`, `message` | The model can't call tools; this turn runs without |
| `turn_error` | `message`, `attempt` | A turn failed and is retried; three in a row end the run |
| `final_answer_nudge` | `max_iters` | Tool turns ran out; the model is asked to answer |
| `shared_model` | `model` | A goal is running, so this chat uses its model |
| `nudge` | `name`, `reason` | A goal's last stage stopped to ask and was sent back |
| `check` | `verdict`, `reason` | The done-when check: met, not met, or unchecked |
| `error` | `message` | The run failed; the last event of its stream |
| `done` | `answer`? | The run finished |
| `goals` | `running`, `pausing`, `current`, `stage`, `awaiting`, `note`, `settings`, `goals` | The whole queue (the snapshot) |
| `agent_start` | `goal`, `name`, `persona`, `model`, `cycle`, `resumed`, `sandbox` | Agent mode: a run begins |
| `cycle` | `n`, `model`, `goal` | Agent mode: a cycle begins |
| `cycle_error` | `n`, `message`, `errors`, `consecutive` | Agent mode: a cycle failed; it backs off and continues |
| `idle` | `seconds` | Agent mode: pausing between cycles |
| `agent_done` | `cycle`, `summary` | Agent mode: the goal was marked done |
| `agent_stop` | `reason`, `cycle`, `detail`, `last_error`? | Agent mode: the loop stopped |
| `agent_end` | `cycle`, `done`, `summary`, `errors`, `stopped_reason` | Agent mode: the last event of a run |

A goal in a snapshot: `id`, `text`, `done_when`, `agents`, `status` (pending, running,
done, failed, skipped), `route` (the stages that ran), `answer`, `error`, `added`,
`started`, `finished`, `check`.

## Examples

`contract/examples/` holds streams recorded from a real `lag serve`, through the phone
port with a device key, as the app will connect:

| File | What it is |
|---|---|
| `chat-tool.jsonl` | A one-agent chat that calls a tool |
| `pipeline-routed.jsonl` | A pipeline where the orchestrator picks the agents |
| `stream-failure.jsonl` | A chat on a model that doesn't exist: failed turns, then the end |
| `goal-run.jsonl` | `/api/goals/events` while a goal is added and run to the end |
| `responses.json` | One reply from each plain endpoint, plus a 401 |

Personal details are scrubbed before saving: the home folder becomes `/Users/you`, the
pairing key is replaced, and the model list keeps only the models the roles use.

## Changing the contract

- **Adding** an event type or a field: add it to `contract/v1.json` and to this page.
  `tests/test_contract.py` fails until both are done.
- **Removing, renaming or retyping**: that breaks clients, so it needs contract 2: a new
  `contract/v2.json`, `CONTRACT_VERSION = 2` in `web.py`, and the app updated to match.
- **After changing what the server sends**, re-record the examples with
  `contract/record.py` (its docstring says how) and check them before committing.
