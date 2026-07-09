*This project has been created as part of the 42 curriculum by nappasam.*

# Call Me Maybe

## Description

**Call Me Maybe** is a constrained-decoding function-calling system built entirely in Python. Its goal is to take natural-language prompts (e.g. *"What is the sum of 2 and 3?"*) and turn them into structured, machine-readable function calls (e.g. `fn_add_numbers(a=2, b=3)`) — without ever letting the model produce free-form or malformed text.

The core challenge this project solves is a well-known limitation of small language models: when simply *prompted* to output JSON, a model like `Qwen/Qwen3-0.6B` will often produce syntactically broken or schema-violating output. Rather than relying on prompting alone, this project implements **constrained decoding**: a finite state machine (FSM) intervenes directly in the token generation loop, masking out every token that would break JSON structure or violate the expected function schema. The result is output that is **100% syntactically valid JSON**, guaranteed by construction rather than by hope.

The system supports five predefined functions:

| Function | Purpose |
|---|---|
| `fn_add_numbers` | Add two numbers |
| `fn_greet` | Generate a greeting |
| `fn_reverse_string` | Reverse a string |
| `fn_get_square_root` | Compute a square root |
| `fn_substitute_string_with_regex` | Regex-based string substitution |

Given a batch of natural-language prompts and a set of function definitions, the program selects the correct function *using the LLM itself* (never heuristics) and extracts the correct arguments, writing everything to a single validated JSON output file.

## Instructions

### Requirements

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management
- The `llm_sdk` package (provided as a workspace dependency in `llm_sdk_source/`)

### Installation

```bash
uv sync
```

This installs all dependencies declared in `pyproject.toml`, including `pydantic`, `numpy`, and the `llm_sdk` workspace package.

### Running the project

```bash
PYTHONPATH=llm_sdk_source uv run python -m src
```

By default the program reads:
- `data/input/function_definitions.json` — the schema of the five available functions
- `data/input/function_calling_tests.json` — the list of prompts to process

and writes its results to:
- `data/output/function_calling_results.json`

Each entry in the output array contains exactly three keys: `prompt`, `name`, and `parameters`.

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

### Makefile targets

| Target | Description |
|---|---|
| `make install` | Installs project dependencies via `uv` |
| `make run` | Runs the main program (`uv run python -m src`) |
| `make debug` | Runs the program under `pdb` for step-through debugging |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, and other build artifacts |
| `make lint` | Runs `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `make lint-strict` | Runs `flake8 .` and `mypy . --strict` |

## Algorithm Explanation

### Why constrained decoding, and not just prompting

A language model generates text one token at a time. At each step it produces a probability distribution (logits) over its entire vocabulary, and normally the highest-probability (or sampled) token is chosen. If you only *prompt* the model to output JSON, nothing stops it from picking a token that breaks JSON syntax or violates the function schema — a 0.6B-parameter model is small enough that this happens often.

Constrained decoding intervenes **before** token selection at every single step:

1. The model produces logits for the full vocabulary.
2. The decoder computes the set of tokens that would keep the output both valid JSON *and* compliant with the current function/argument schema.
3. Every other token's logit is set to `-inf`.
4. The next token is chosen only from what remains.

This repeats token-by-token until the full `{"name": ..., "parameters": {...}}` object has been generated, guaranteeing the result is always parseable — never "usually."

### The pipeline

```
prompt.json ─▶ parser.py ─▶ promptbuilder.py ─▶ decoder.py ─▶ output.py ─▶ results.json
 (validate)      (validate      (build one-shot     (token-by-token       (write &
                  schema)         prompt per          FSM-constrained       validate)
                                  request)             generation)
```

- **`parser.py`** loads and validates `function_definitions.json` and `function_calling_tests.json` against pydantic models, so malformed input is rejected before generation even starts.
- **`promptbuilder.py`** constructs the natural-language prompt sent to the model for each request, including a dynamically-built one-shot example (see *Design Decisions*).
- **`decoder.py`** is the heart of the project: it drives the FSM that constrains generation.
- **`output.py`** assembles and writes the final validated JSON.

### The finite state machine

The decoder models JSON generation as a sequence of states, e.g. `EXPECTING_NAME → INSIDE_FUNCTION_NAME → EXPECTING_PARAMETERS → EXPECTING_OPEN_PARAMETER_BRACE → EXPECTING_PARAMETER_KEY → EXPECTING_COLON → INSIDE_PARAMETER_VALUE_{NUMBER,STRING,BOOLEAN} → EXPECTING_COMMA → EXPECTING_CLOSING_PARAMETER_BRACE`.

At every state, the set of *valid next tokens* is computed dynamically from the vocabulary JSON exposed by the SDK (`get_path_to_vocabulary_json`) — not precomputed once and reused, because a precomputed union over all possible function/parameter names quickly leads to invalid intermediate states and infinite loops. Instead, valid tokens are **prefix-aware**: only tokens consistent with what has already been generated in the current state are allowed. For example, while inside `INSIDE_PARAMETER_VALUE_NUMBER`, only digit tokens, a single decimal point, and the closing delimiters (`,` / `}`) are valid; once a value is closed, control transitions back to `EXPECTING_PARAMETER_KEY` or `EXPECTING_COMMA` depending on context.

Function names and parameter keys are matched by accumulating generated tokens and comparing them against pre-encoded token sequences for every candidate name/key; once a full match is found, the FSM commits to that choice and transitions state.

### Performance optimization: the single-token fast path

The single biggest optimization in this project is short-circuiting the model's forward pass whenever the valid-token set has exactly one member. If only one token can possibly be correct at a given position (e.g. the closing `"` after a function name has been fully matched), there is no need to run the model at all — the token is simply appended. This fast path was responsible for cutting total runtime from roughly 12 minutes down to roughly 3.5 minutes across the full test batch.

## Design Decisions

- **Control flow via return values, not exceptions.** The parser follows a consistent `None`-return convention for "no match" / invalid states. Exceptions are reserved for genuinely exceptional conditions, not for expected control flow — this keeps error handling predictable and avoids catch-all `except` blocks silently swallowing real bugs.
- **Dynamic, prefix-aware token sets.** As described above, valid tokens are computed fresh at each generation step rather than precomputed statically, which is what makes the FSM correct across arbitrary function/parameter names instead of just the ones seen during development.
- **Prompt isolation.** `self.prompt` holds the *entire* constructed prompt, including the one-shot example — which itself contains numbers and text that could be mistaken for the user's actual request if scanned naively. `self.current_request`, derived via `rfind('Request: "')`, isolates only the real user request, preventing the one-shot example from leaking into intent detection.
- **One-shot prompt design.** The prompt does not include a `Returns:` field (which was empirically found to bias the model toward defaulting to the wrong function), and instead builds a one-shot example dynamically from `function_definitions[0]`, framing the actual task as `Request: "..."` / `Call:`. This measurably eliminated a class of wrong-function-selection errors.
- **Function selection is left to the LLM.** No heuristics or keyword matching are used to pick which function to call — per the subject's requirements, function selection is entirely the model's responsibility, constrained only in *how* it can express that choice.

## Performance Analysis

- **Accuracy:** 10/11 on the private test set, 9/11 on the public test set — both above the 90% accuracy threshold required by the subject for function selection and argument extraction.
- **Speed:** Full test batches complete in well under the 5-minute ceiling required by the subject, primarily thanks to the single-token fast path described above (~12 min → ~3.5 min).
- **Reliability:** Because the FSM masks invalid tokens at the logit level rather than validating after the fact, output JSON is syntactically valid 100% of the time by construction — validity is not something that can regress with different prompts.
- **Known ceiling:** One test case (escaped-quote handling inside string parameter values) cannot pass under the current FSM design. The state machine's string-value state does not currently model escape sequences as part of its transition logic, so a string value containing an escaped quote cannot be generated correctly. This is an architectural limitation of the current state design rather than a bug, and is documented here rather than silently left unexplained.

## Challenges Faced

- **Precomputed token sets caused infinite loops.** An early version computed a single static union of valid tokens across all function/parameter names. Because this union didn't reflect what had already been generated (prefix state), the decoder could get stuck oscillating between invalid partial matches. This was solved by making token-set computation dynamic and prefix-aware at every step.
- **Prompt pollution from the one-shot example.** Early intent-detection logic scanned the full constructed prompt, which meant embedded numbers/text from the one-shot example were sometimes mistaken for the actual user request. Isolating `self.current_request` via `rfind('Request: "')` fixed this.
- **Wrong-function defaults.** Including a `Returns:` field in the prompt biased the model toward defaulting to a fixed function regardless of the actual request. Removing it and reframing the prompt as `Request:` / `Call:` resolved this.
- **Remaining edge-case crashes**, tracked as known issues rather than silently hidden:
  - An empty `[]` for `function_definitions` triggers an `IndexError` in `promptbuilder.py`.
  - An empty `[]` for `function_calling_tests` triggers a `ZeroDivisionError` in `__main__.py`.
  - Parameter types outside `number`/`string`/`boolean` (e.g. `array`, `object`) are not currently modeled by the FSM and can silently loop rather than erroring cleanly.
  - Adversarial fuzzing of the numeric parameter state produces a crash in roughly 32% of runs, indicating the number-value transition logic doesn't yet cover every malformed numeric edge case.
- **Escaped-quote handling** (see *Performance Analysis*) remains an open architectural limitation rather than a fixed bug.

## Testing Strategy

Validation was primarily empirical rather than purely theoretical: implementation choices were checked by running actual batches of test prompts and inspecting real terminal output, since FSM behavior over token streams is difficult to fully reason about in the abstract.

- **Print-based tracing** was the primary debugging tool — logging the current FSM state, the accumulated tokens, and the computed valid-token set at each generation step made it possible to catch incorrect state transitions directly.
- **The moulinette** (the project's automated grader) runs the program against private and public test sets, calling the real Python functions with extracted arguments and comparing return values against expected outputs — this is the authoritative accuracy measure reported above.
- **Adversarial/fuzz testing** was used specifically to stress-test the numeric parameter state and edge cases like empty input lists, which is how the remaining known bugs (listed above) were identified.
- Code quality was checked with `flake8` (79-character line length) and `mypy` in strict mode, run against `parser.py`, `output.py`, `decoder.py`, and `src/__main__.py`.

## Example Usage

```bash
# 1. Install dependencies
uv sync

# 2. Run against the default input files
PYTHONPATH=llm_sdk_source uv run python -m src

# 3. Inspect the results
cat data/output/function_calling_results.json
```

Example input (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

Example output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Resources

### References

- [OpenAI Function Calling documentation](https://platform.openai.com/docs/guides/function-calling) — general conceptual background on structured function calling with LLMs.
- [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation ("Outlines")](https://arxiv.org/abs/2307.09702) — background reading on logit-masking approaches to constrained generation.
- [JSON Schema specification](https://json-schema.org/) — reference for schema/validation vocabulary used to reason about the function definitions.
- [Qwen3 model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B) — documentation for the model used in this project.
- Byte-Pair Encoding (BPE) tokenization references, for understanding how the vocabulary JSON maps tokens to strings.

### Use of AI

Claude (Anthropic) was used throughout development as a collaborative debugging and design-review partner, not as a code generator for the core logic:

- **Architecture and FSM design discussion** — reasoning through state transitions, the prefix-aware vs. static token-set problem, and why single-token fast-pathing works, primarily through guided/Socratic back-and-forth rather than being handed a finished design.
- **Debugging support** — interpreting print-trace output and terminal results to isolate the root cause of bugs (e.g. the prompt-pollution issue, the wrong-function-default issue).
- **Code quality passes** — assistance bringing `parser.py`, `output.py`, `decoder.py`, and `src/__main__.py` into `flake8`/`mypy`-strict compliance without changing behavior.
- **Documentation** — this README was drafted with AI assistance, based on the actual implementation and the project subject's requirements.

All core algorithmic decisions (the FSM design, the prompt structure, the fast-path optimization) were made and implemented by the author; AI assistance was used for review, debugging, and polish rather than for generating the constrained-decoding logic itself.
