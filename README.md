# HuggingFace Inference CLI for Story Writing

Minimal Node.js CLI for hosted LLM story generation.

Features:

-   Separate `context` and `instruct` inputs
-   Chat‑style sessions (rolling context)
-   Auto‑continue across provider caps
-   File inputs, stdout or file output
-   Provider routing (Together, Fireworks AI)
-   PowerShell‑friendly

## Requirements

-   Node.js 18+
-   Hugging Face token with inference enabled
-   File `.env.local` at project root containing:

```text
HF_TOKEN_CLI=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

## Install

```bash
npm install @huggingface/inference
```

## Quick start

### One‑shot

```bash
node ./index.mjs mistralai/Mistral-7B-Instruct-v0.3 context "../prompts/premise.txt" instruct "Write a complete short story. 1200–1600 words." --out ../out/story1.md
```

### Start a session

```bash
node ./index.mjs mistralai/Mistral-7B-Instruct-v0.3 --session ../sessions/hero.jsonl --reset context "../prompts/premise.txt" instruct "Begin the story. Introduce the protagonist and setting."
```

### Continue the same story

```bash
node ./index.mjs mistralai/Mistral-7B-Instruct-v0.3 --session ../sessions/hero.jsonl instruct "Continue to Chapter 2. The protagonist is forced to choose." --out ../out/chapter2.md
```

### Longer scenes (auto‑continue)

```bash
node ./index.mjs mistralai/Mistral-7B-Instruct-v0.3 --session ../sessions/arc.jsonl instruct "Write the midpoint confrontation. 1800–2200 words." --chunks 1400 --target 2200
```

> ```bash
> context "@../prompts/premise.txt"
> ```

### File path compatibility note

> **Important:** To use a file as input for `context` or `instruct`, you **must** prefix the file path with `@` (e.g., `@../prompts/foo.txt`). If you do not use the `@` prefix, the argument will be treated as a literal string, not a file path.
>
> ```bash
> context "@../prompts/premise.txt"
> instruct "@../prompts/instruct.txt"
> ```

> PowerShell note: always quote file paths, e.g. "@../prompts/foo.txt".

## Usage

```text
node ./index.mjs <model_id> [context <text|file>] [instruct <text|file>]
[--session <file>] [--reset]
[--out <file>]
[--max <tokens>] [--chunks <tokens>] [--target <words>]
[--provider <name>]
[--debug]
```

## Flags

| Flag         | Purpose                                                 |
| ------------ | ------------------------------------------------------- |
| `context`    | World, characters, tone. Accepts a string or file path  |
| `instruct`   | The action for this turn. Accepts a string or file path |
| `--session`  | JSONL chat log file for rolling context                 |
| `--reset`    | Start a fresh session                                   |
| `--out`      | Write output to a file                                  |
| `--max`      | Per‑response token cap (default: 1200)                  |
| `--chunks`   | Per‑chunk cap when auto‑continuing (default = `--max`)  |
| `--target`   | Target word count across stitched chunks (0 disables)   |
| `--provider` | Force provider (e.g., `together`, `fireworks-ai`)       |
| `--debug`    | Verbose logs (finish reasons, usage, session load/save) |

## Concepts

### Context vs Instruction

-   `context`: background, lore, constraints, style, POV
-   `instruct`: the actionable request for this turn

Both accept raw strings or file paths.

### Sessions

Use `--session` to persist the message history so each call “remembers” prior output.  
Use `--reset` to start over.  
Session files are JSONL (one JSON object per line).

### Auto‑continue

Providers clamp response length. If a reply hits the cap (`finish_reason = "length"`), the CLI issues a “Continue.” turn and stitches the next chunk.

Use:

-   `--chunks`: tokens per chunk (e.g., 1200–1600)
-   `--target`: desired final word count across chunks

## Story patterns

### Noir opener

```bash
instruct "Write a 1400-word gritty noir scene in a rain-soaked alley. Tight third-person, cynical tone, sensory detail. End on a hook."
```

### Horror escalation

```bash
context "../prompts/lore.txt"
instruct "Escalate dread without gore. Focus on sound and shadow. ~1500 words."
```

### Romance or adult themes

> Hosted models may filter explicit detail. Steering still helps.

```bash
instruct "Write a romantic, sensual scene between consenting adults. Emphasize interiority, consent, and atmosphere; avoid clinical terms. 1200–1600 words."
```

### Choose‑your‑own‑adventure beat

```bash
instruct "Write a scene that ends with two numbered options for the reader. Each option should clearly branch the story."
```

### Iterative revision

```bash
instruct "Rewrite the previous scene with subtler menace, less exposition, and stronger subtext. Keep POV and continuity."
```

## Length control

-   Ask explicitly for words: e.g., “1300–1700 words”
-   Use `--chunks 1200 --target 2000` for stitched long scenes
-   If it ends early, run again with the same session and instruct `"Continue the scene."`

## Tested model/provider pairs

Make sure to accept model licenses on huggingface.co before using.

-   `mistralai/Mistral-7B-Instruct-v0.3` — `--provider together`
-   `meta-llama/Llama-3.1-8B-Instruct` — `--provider fireworks-ai`
-   `Qwen/Qwen2.5-7B-Instruct` — `--provider together`
-   `google/gemma-2-9b-it` — `--provider fireworks-ai`

## Directory layout

```text
project-root/
├── .env.local
├── cli/
│ └── index.mjs
├── prompts/
│ └── premise.txt
├── sessions/
│ └── my-story.jsonl
├── out/
│ └── ch1.md
└── README.md
```

## Troubleshooting

-   `ENOENT` on file: paths resolve relative to `./cli/` and `../`. Use "../prompts/foo.txt" or absolute path.
-   `No Inference Provider available`: model isn’t served by your providers. Try `--provider together` or change model.
-   License errors: visit the model card on Hugging Face and click "Accept".
-   Short replies despite `--max`: use `--chunks` and `--target`, or follow up with a `Continue` turn.
-   PowerShell quoting: always quote file paths, e.g. "../prompts/foo.txt".

## Roadmap

-   `--summarize` to compact early chapters into “Story So Far”
-   `--outdir` with auto‑numbered files
-   Web UI version that reuses session format

## License

MIT. Use responsibly.
