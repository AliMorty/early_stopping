# Project Rules

- When running a Bash command that requires user approval, always include a brief plain-English explanation of what the command does before the tool call, so the user knows what they are approving.
- When the user asks to end or summarize the session, create a markdown file in `session_summaries/` summarizing what was done during the session at a high level. Name it with the date (e.g. `2026-03-02.md`). If multiple sessions happen on the same day, append a number (e.g. `2026-03-02_2.md`). This helps trace back ideas and discussions across sessions.

## When You Hit a Problem
- If something doesn't work as expected (API returns no data, a plan hits a dead end, an assumption turns out to be wrong), **stop and tell the user immediately**.
- Do NOT silently try ad-hoc workarounds or pivot to a different approach without informing the user first.
- Clearly state: what you tried, what went wrong, and what the options are — then wait for direction.
