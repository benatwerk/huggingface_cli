// ./cli/index.mjs
import fs from "node:fs";
import path from "node:path";
import { config } from "dotenv";
import { InferenceClient } from "@huggingface/inference";

// Load .env.local specifically
config({ path: path.resolve(process.cwd(), ".env.local") });

/* ---------------- env: load .env.local (HF_TOKEN_CLI=hf_xxx) --------------- */
if (!process.env.HF_TOKEN_CLI) {
    console.error("Missing HF_TOKEN_CLI in .env.local");
    process.exit(1);
}

/* --------------------------------- helpers ----------------------------------- */
const readArg = (a, debug = false) => {
    if (!a) return "";
    if (!a.startsWith("@")) return a;
    const raw = a.slice(1);
    const tries = [
        path.resolve(process.cwd(), raw), // ./cli/<raw>
        path.resolve(process.cwd(), "..", raw), // ../<raw>
        path.isAbsolute(raw) ? raw : null,
    ].filter(Boolean);
    for (const p of tries) {
        if (fs.existsSync(p)) {
            // Always return the file path in debug mode
            return debug ? `@${p}` : fs.readFileSync(p, "utf8");
        }
    }
    throw new Error(`File not found for ${a}. Tried:\n${tries.join("\n")}`);
};

const takeFlag = (args, name) => {
    const i = args.indexOf(name);
    if (i === -1) return null;
    const v = args[i + 1];
    args.splice(i, 2);
    return v;
};

const popSwitch = (args, name) => {
    const i = args.indexOf(name);
    if (i === -1) return false;
    args.splice(i, 1);
    return true;
};

/* === Session helpers === */
const loadSession = (p, debug = false) => {
    try {
        if (!p || !fs.existsSync(p)) return [];
        const lines = fs.readFileSync(p, "utf8").split(/\r?\n/).filter(Boolean);
        const msgs = lines.map((l) => JSON.parse(l));
        if (debug)
            console.error(
                `[debug] session loaded: ${msgs.length} turns from ${p}`
            );
        return msgs;
    } catch (e) {
        console.error(`[debug] failed to load session ${p}:`, e.message);
        return [];
    }
};

const saveSession = (p, msgs, debug = false) => {
    try {
        if (!p) return;
        fs.mkdirSync(path.dirname(p), { recursive: true });
        fs.writeFileSync(
            p,
            msgs.map((m) => JSON.stringify(m)).join("\n") + "\n",
            "utf8"
        );
        if (debug)
            console.error(`[debug] session saved: ${msgs.length} turns → ${p}`);
    } catch (e) {
        console.error(`[debug] failed to save session ${p}:`, e.message);
    }
};

// sliding window by character budget
const packMessages = (arr, maxChars = 140000, debug = false) => {
    let total = 0,
        out = [];
    for (let i = arr.length - 1; i >= 0; i--) {
        total += arr[i].content?.length || 0;
        out.push(arr[i]);
        if (total > maxChars) break;
    }
    out = out.reverse();
    if (debug)
        console.error(
            `[debug] packed history: ${out.length} turns, ~${total} chars`
        );
    return out;
};

/* ------------------------------ CLI parsing ---------------------------------- */
const argv = process.argv.slice(2);
if (argv.length < 1) {
    console.error(`usage:
  node ./index.mjs <model_id> [context <text|@file>] [instruct <text|@file>]
                   [--out <file>] [--max <n>] [--target <words>] [--chunks <n>]
                   [--provider <name>] [--session <file>] [--reset] [--debug]

examples:
  node ./index.mjs mistralai/Mistral-7B-Instruct-v0.3 --session ../sessions/s1.jsonl --reset context "@../prompts/premise.txt" instruct "Open scene."
  node ./index.mjs meta-llama/Llama-3.1-8B-Instruct instruct "@../prompts/scene.txt" --out ../out/scene.md --max 1200 --provider fireworks-ai
`);
    process.exit(1);
}

const model = argv.shift();
const outPath = takeFlag(argv, "--out");
const maxTokens = Number(takeFlag(argv, "--max") ?? 1200); // per response
const targetWords = Number(takeFlag(argv, "--target") ?? 0); // 0 disables target
const chunkMax = Number(takeFlag(argv, "--chunks") ?? maxTokens);
const providerOverride = takeFlag(argv, "--provider");
const sessionPath = takeFlag(argv, "--session");
const resetSession = popSwitch(argv, "--reset");
const debug = popSwitch(argv, "--debug");

// positional pairs: context <val>, instruct <val>
let contextText = "";
let instructText = "";
for (let i = 0; i < argv.length; ) {
    const key = argv[i];
    if (key === "context" || key === "instruct") {
        const val = argv[i + 1];
        if (!val) {
            console.error(`Missing value after ${key}`);
            process.exit(1);
        }
        const content = readArg(val, debug); // Pass debug flag here
        if (key === "context") contextText = content;
        else instructText = content;
        i += 2;
    } else {
        i += 1; // ignore stray tokens
    }
}
if (!contextText && !instructText && !sessionPath) {
    console.error(
        "Provide at least one of: context, instruct, or a --session with prior turns."
    );
    process.exit(1);
}

/* --------------------------- Prompt + session state -------------------------- */
const userPrompt = [
    contextText ? `### Context\n${contextText.trim()}` : null,
    instructText ? `### Instruction\n${instructText.trim()}` : null,
]
    .filter(Boolean)
    .join("\n\n");

let messages = resetSession ? [] : loadSession(sessionPath, debug);
messages = packMessages(messages, 140000, debug);

if (userPrompt?.trim()) {
    messages.push({ role: "user", content: userPrompt });
    if (debug)
        console.error(`[debug] appended user turn (len=${userPrompt.length})`);
}
if (messages.length === 0) {
    messages.push({ role: "user", content: "Continue the story." });
    if (debug)
        console.error("[debug] seeded default user turn for continuation");
}

/* ---------------------------- Inference plumbing ----------------------------- */
const providers = providerOverride
    ? [providerOverride]
    : ["huggingface"]; // Use "huggingface" as default instead of empty array
const hf = new InferenceClient(process.env.HF_TOKEN_CLI);

// auto-continue across provider caps
async function autoContinueChat({
    model,
    provider,
    baseMessages,
    chunkMax,
    targetWords,
    maxRounds = 6,
    debug = false,
}) {
    let msgs = baseMessages.slice();
    let out = "";

    if (debug) {
        console.error("[debug] Debug mode enabled. Request metadata:");
        console.error(
            JSON.stringify(
                {
                    model,
                    provider: provider || "huggingface",
                    max_tokens: chunkMax,
                    temperature: 0.85,
                    top_p: 0.92,
                    extra_body: { max_output_tokens: chunkMax },
                },
                null,
                2
            )
        );
        return { text: "(debug mode: no request sent)", messages: msgs };
    }

    for (let round = 0; round < maxRounds; round++) {
        const requestBody = {
            model,
            // Only pass provider if it's not "huggingface"
            ...(provider && provider !== "huggingface" && { provider }),
            messages: msgs,
            max_tokens: chunkMax,
            temperature: 0.85,
            top_p: 0.92,
        };

        // Only add extra_body for providers that support it
        if (provider && provider !== "huggingface") {
            requestBody.extra_body = { max_output_tokens: chunkMax };
        }

        const res = await hf.chatCompletion(requestBody);

        const piece = res?.choices?.[0]?.message?.content ?? "";
        const finish = res?.choices?.[0]?.finish_reason ?? "";
        const usage = res?.usage;

        if (debug)
            console.error(
                `[debug] round=${round} provider=${provider} finish=${finish} usage=${JSON.stringify(
                    usage || {}
                )}`
            );
        out += (out ? "\n\n" : "") + piece;

        msgs.push({ role: "assistant", content: piece });

        if (finish !== "length") break;
        if (targetWords && out.trim().split(/\s+/).length >= targetWords) break;

        msgs.push({ role: "user", content: "Continue." });
    }

    return { text: out, messages: msgs };
}

/* --------------------------- Debug Output Section ---------------------------- */
if (debug) {
    console.error("[debug] Debug mode enabled. Parameters and settings:");
    console.error(
        JSON.stringify(
            {
                model,
                outPath: outPath || "(not specified)",
                maxTokens,
                targetWords,
                chunkMax,
                providerOverride: providerOverride || "(default providers)",
                sessionPath: sessionPath || "(not specified)",
                resetSession,
                contextText: contextText.startsWith("@")
                    ? contextText
                    : "(inline text provided)",
                instructText: instructText.startsWith("@")
                    ? instructText
                    : "(inline text provided)",
                providers,
            },
            null,
            2
        )
    );

    // Exit immediately after showing debug info
    process.exit(0);
}

/* --------------------------------- run call ---------------------------------- */
console.error(`[debug] About to try providers: ${JSON.stringify(providers)}`);
console.error(`[debug] Model: ${model}`);
console.error(`[debug] HF Token: ${process.env.HF_TOKEN_CLI ? 'SET' : 'NOT SET'}`);

let lastErr;
for (const provider of providers) {
    console.error(`[debug] Trying provider: ${provider}`);
    try {
        const baseMessages = messages;

        console.error(`[debug] Calling autoContinueChat with provider: ${provider}`);
        const { text, messages: newMsgs } = await autoContinueChat({
            model,
            provider,
            baseMessages,
            chunkMax,
            targetWords,
            maxRounds: 6,
            debug,
        });

        const finalText = text || "(no content)";
        if (debug)
            console.error(
                `[debug] final output (${finalText.length} chars):\n${finalText}`
            );

        // For Markdown output, ensure single newlines between paragraphs
        const mdSafeText = finalText
            .replace(/\n{3,}/g, "\n\n") // Reduce multiple newlines to two
            .replace(/([^\n])\n([^\n])/g, "$1 $2"); // Prevent single line breaks between words

        if (outPath) {
            fs.writeFileSync(outPath, mdSafeText, "utf8");
            console.log(`Wrote output to ${outPath}`);
        } else {
            console.log(mdSafeText);
        }

        saveSession(sessionPath, newMsgs, debug);
        process.exit(0);
    } catch (e) {
        lastErr = e;
        console.error(
            `Error with provider ${provider} (model=${model}): ${e.message}`
        );
        console.error(`[debug] Full error:`, e);
        console.error(`[debug] Error stack:`, e.stack);
    }
}

// Fix the error handling
if (lastErr) {
    console.error(
        `All providers failed. Giving up. Last error: ${lastErr.message}`
    );
} else {
    console.error("No providers available to try.");
}
process.exit(1);
