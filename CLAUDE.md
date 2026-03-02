# Project Rules

- When running a Bash command that requires user approval, always include a brief plain-English explanation of what the command does before the tool call, so the user knows what they are approving.
- When the user asks to end or summarize the session, create a markdown file in `session_summaries/` summarizing what was done during the session at a high level. Name it with the date (e.g. `2026-03-02.md`). If multiple sessions happen on the same day, append a number (e.g. `2026-03-02_2.md`). This helps trace back ideas and discussions across sessions.
