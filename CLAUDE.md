# Project Rules

- When running a Bash command that requires user approval, always include a brief plain-English explanation of what the command does before the tool call, so the user knows what they are approving.
- When the user asks to end or summarize the session, create a markdown file in `session_summaries/` summarizing what was done during the session at a high level. Name it with the date (e.g. `2026-03-02.md`). If multiple sessions happen on the same day, append a number (e.g. `2026-03-02_2.md`). This helps trace back ideas and discussions across sessions. Sign the summary at the bottom with the model that wrote it (e.g. `— Claude Sonnet 4.6` or `— Claude Opus 4.6`).
- After writing the session summary, also update `project_state/PROJECT_STATE.md` to reflect the current state: mark completed items, add new next steps, update open questions, and note any new known bugs or design decisions.

## Do Not Change Things Unless Asked
- Only modify what the user explicitly requests. Do not resolve open questions, change design decisions, or update things "while you're at it" without asking first.
- If you notice something that could be improved or a decision that could be resolved, flag it and ask — do not act on it unilaterally.

## Protected Algorithm Code — NEVER Edit (protection by location)
- **Any file directly in `ali_code/` (its root) is Ali's own code. NEVER modify it** — not a one-line change, "obvious" fix, or refactor. This includes the algorithm classes, notebooks, and any generated module artifact sitting in the root.
- **LLM-generated / assistant-owned files live in `LLM_*` subfolders of `ali_code/`** (e.g. `ali_code/LLM_visualization/`). You may freely create and edit files there.
- If a task appears to require changing protected (root) code, **stop and describe the exact edit for the user to make themselves** (quote the lines and the change). Do not apply it. Wait for the user to make the change and confirm. Prefer solving the problem in the `LLM_*` folder instead; only surface a protected-code change as a last resort suggestion.
- Regenerating a derived artifact (e.g. running `build_logistic_module.py` to rebuild `logistic_regression.py` from the notebook) is allowed — that is generation, not hand-editing the source.

## Be Careful When Deleting or Testing
- **Never `rm`, overwrite, or otherwise erase files in directories that hold the user's data, caches, or outputs** (e.g. `ali_code/LLM_visualization/logistic_cache/`, saved `.pkl` runs, `index.json`, `last_params.json`). These may contain results the user created and cannot easily regenerate.
- **When you need scratch space to test something, create and use a separate throwaway directory** (e.g. under `/tmp/`, or a clearly-named `*_tmp/` folder) — never the live data directory. Point test caches/outputs at the temp dir. Clean up your own temp dir, not the user's.
- **Before any deletion or overwrite, confirm what you are removing is something you created for the test**, not user data. If unsure, stop and ask.

## Token Efficiency
- Prefer writing and running scripts over predicting output by hand. Write code, execute it, and use the actual result.
- Avoid loading large data into context unnecessarily — let scripts handle it instead.

## When You Hit a Problem
- If something doesn't work as expected (API returns no data, a plan hits a dead end, an assumption turns out to be wrong), **stop and tell the user immediately**.
- Do NOT silently try ad-hoc workarounds or pivot to a different approach without informing the user first.
- Clearly state: what you tried, what went wrong, and what the options are — then wait for direction.

## Code Reviews Must Be Thorough
- When asked to review code, act as a serious reviewer. The goal is to find errors, not to confirm that things look fine.
- For every function and every file: understand what it is supposed to do (from the spec, paper, or design docs), then verify that it actually does that thing correctly in the general case — not just for the default or happy-path inputs.
- Do not hide bugs behind disclaimers like "safe for the default config" or "this is also a violation of assumptions." If a function does not correctly implement what it claims to implement, that is a bug. State it clearly.
- Do not flag a real issue in a footnote and move on. If something is wrong, it belongs at the top of the findings, with a clear explanation of what breaks and when.
- Check edge cases and general inputs, not just the specific values used in the current experiments.
- Be honest. If something looks wrong, say it clearly even if you are not 100% certain. Do not soften findings to avoid conflict.
- If you do not know something or cannot verify a claim, say so explicitly. Do not paper over uncertainty with vague qualifiers.
- Share concerns proactively. If something worries you during a review, explain it fully even if not directly asked. The user needs the full picture.
