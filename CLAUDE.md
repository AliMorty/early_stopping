# Project Rules

- When running a Bash command that requires user approval, always include a brief plain-English explanation of what the command does before the tool call, so the user knows what they are approving.
- When the user asks to end or summarize the session, create a markdown file in `session_summaries/` summarizing what was done during the session at a high level. Name it with the date (e.g. `2026-03-02.md`). If multiple sessions happen on the same day, append a number (e.g. `2026-03-02_2.md`). This helps trace back ideas and discussions across sessions. Sign the summary at the bottom with the model that wrote it (e.g. `— Claude Sonnet 4.6` or `— Claude Opus 4.6`).
- After writing the session summary, also update `project_state/PROJECT_STATE.md` to reflect the current state: mark completed items, add new next steps, update open questions, and note any new known bugs or design decisions.

## Do Not Change Things Unless Asked
- Only modify what the user explicitly requests. Do not resolve open questions, change design decisions, or update things "while you're at it" without asking first.
- If you notice something that could be improved or a decision that could be resolved, flag it and ask — do not act on it unilaterally.

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
