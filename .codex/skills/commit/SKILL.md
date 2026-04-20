---
name: commit
description: Stage the relevant git changes and create a commit using this repo's emoji-prefixed conventional commit format. Use when the user asks Codex to commit changes, stage and commit selected files, or draft a concise Chinese commit message from the current diff.
---

# Commit

Create a focused git commit that stages only the requested changes and follows this repo's emoji + conventional commit style.

## Workflow

1. Inspect the current state with `git status --short` and the relevant diff with `git diff`, `git diff --cached`, or both.
2. Determine the commit scope before staging.
   - Stage only files related to the requested task.
   - Prefer explicit file paths over `git add -A`.
   - If the worktree contains unrelated changes and the scope is ambiguous, pause and confirm before staging.
3. Choose the commit type and matching emoji.
4. Write a concise Chinese subject with the format `<emoji> <type>: <description>`.
5. Add AI attribution only when the user explicitly asks for it or the repo already has a clear trailer convention.
   - Reuse the exact trailer requested by the user or already present in repo history.
   - Do not invent a new `Co-Authored-By` identity when no convention exists.
6. Run `git commit` non-interactively and report what was included.

## Emoji Mapping

- `✨ feat`: new feature
- `🐛 fix`: bug fix
- `📝 docs`: documentation changes
- `💄 style`: formatting, UI, or cosmetic changes
- `♻️ refactor`: code cleanup or restructuring without behavior change
- `⚡ perf`: performance improvement
- `✅ test`: add or update tests
- `🔧 chore`: tooling, config, or maintenance work
- `🚀 deploy`: deployment-related change
- `🔥 remove`: remove code or files

## Message Rules

- Use Chinese by default.
- Keep the subject concise and specific.
- Prefer a single subject line unless extra body text is genuinely useful.
- If the user provided a commit hint, use it to guide the description, but still verify it against the diff.
- Match the actual scope of the staged changes and avoid over-claiming.

## Git Safety

- Never stage unrelated changes just to make the worktree clean.
- Do not amend existing commits unless the user explicitly asks.
- Prefer non-interactive git commands.
- If no relevant changes are ready, explain why instead of forcing a commit.
