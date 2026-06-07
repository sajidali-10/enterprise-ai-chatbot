# AGENTS.md — Development Rules for Enterprise AI Chatbot

> This file defines strict development rules for AI agents working on this project.  
> Violations must be treated as blockers and corrected immediately.

---

## 1. Phased Development

- **Build phase by phase.** Only implement the phase currently in scope.
- **Do not implement future phases unless approved.** Do not add scaffolding, stubs, or TODOs for work not yet requested.
- **Keep the app runnable after every change.** Every commit must leave the application in a working state.

---

## 2. Security & Data Safety

- **Do not hardcode secrets.** No API keys, passwords, tokens, or credentials in source code.
- **Do not connect to production systems.** Local development only against Docker Compose services.
- **Do not use real customer data.** Use synthetic or anonymized data for all development and testing.

---

## 3. Technology Stack

- **Docker Compose** for local development infrastructure.
- **FastAPI** for the backend.
- **Next.js + TypeScript + Tailwind CSS** for the frontend.
- **PostgreSQL** for metadata storage.
- **Qdrant** for vector search.
- **MinIO** for object/file storage.
- **Redis** for caching and job queues.

---

## 4. Explicitly Defer

Do **not** introduce the following until explicitly requested by the user:

- LangChain or LangGraph
- Zendesk integration
- Local LLM deployment (e.g. Ollama, llama.cpp, vLLM)

---

## 5. Task Completion Discipline

- **Show changed files after every task.** Always summarize which files were modified, created, or deleted.
- **Run validation/tests when possible.** Before marking a task complete, run the relevant test suite, linter, or type checker and report the results.

---

## 6. Git Hygiene

- **Stage files explicitly by name** (e.g. `git add path/to/file`).  
  Avoid `git add -A` and `git add .` — they sweep up untracked secrets, build artefacts, and stray files outside the change.
- **Never run destructive commands** (`git reset --hard`, `git push --force`, `git branch -D`, `git clean -f`) on `main`, `master`, `release/*`, or other protected branches without explicit user approval.
- **Prefer creating new commits over amending published commits.** Only amend when the user explicitly asks.
- **Never skip hooks** (`--no-verify`) or bypass signing unless the user explicitly asks. If a hook fails, fix the underlying issue.
- **When running automated git commands that may invoke an editor** (e.g. `git rebase`, `git commit`, `git merge --squash`), set `GIT_EDITOR=true` — an interactive shell must not block execution or cause the command to hang.
- **Do not hardcode branch names like `main` or `master`.** Detect the default branch dynamically (e.g. `git symbolic-ref refs/remotes/origin/HEAD --short | sed 's/origin\///'`). Use the detected name in scripts and commands.

---

## 7. GitHub CLI (`gh`)

`gh` is the canonical interface for GitHub. Prefer it over scraping web URLs or guessing API paths. Discover flags with `gh <cmd> --help` rather than enumerating here.

### Auth
`gh auth status`. If logged out, ask the user to run `gh auth login`.

### Repo
Inferred from cwd. Pass `-R OWNER/REPO` when outside the repo.

### PR Review — Non-Obvious Bits

Find PRs awaiting your review:  
```bash
gh pr list --search "review-requested:@me"
```

Existing review state — two endpoints, easy to confuse:
```bash
gh api repos/OWNER/REPO/pulls/123/comments  --paginate   # inline, line-anchored
gh api repos/OWNER/REPO/issues/123/comments --paginate   # PR-level conversation
```

Post inline comments in one review (line-anchored, multi-comment) — no `gh pr review` flag for this; use the API:
```bash
gh api repos/OWNER/REPO/pulls/123/reviews -f event=COMMENT \
  -f body="overall notes" \
  -F 'comments[][path]=src/foo.py' -F 'comments[][line]=42' \
  -F 'comments[][body]=this is wrong because…'
```

Resolve a review thread (GraphQL):
```bash
gh api graphql -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}' -F id=THREAD_NODE_ID
```

Reply to a specific inline thread:
```bash
gh api repos/OWNER/REPO/pulls/123/comments/COMMENT_ID/replies -f body="fixed in abc1234"
```

Top-level review verbs:  
```bash
gh pr review <N> --approve|--request-changes|--comment -b "…"
```
(See **Consent** section before posting.)

### Workflow Runs

Default to **failed-only** logs, never the full log:
```bash
gh run view 123456 --log-failed          # preferred
gh run view 123456 --log | tail -200     # only if --log-failed isn't enough
```

Find the run behind a PR's latest push:  
```bash
gh pr checks 123 --json name,state,link,workflow
```

### `gh api` Cheatsheet

| Flag | Meaning |
|------|---------|
| `-f key=val` | String parameter |
| `-F key=val` | Typed parameter (numbers, booleans, `@file`) |
| `-X METHOD` | HTTP verb override |
| `--jq '.field'` | Filter response with jq |
| `--paginate` | Follow `Link` headers |

### Never Without Explicit Consent

Anything that publishes, mutates, or notifies needs an explicit in-conversation request. Do **not** run these unprompted:

- **Posting to a PR/issue:** `gh pr review` (any of `--approve`, `--request-changes`, `--comment`), `gh pr comment`, `gh issue comment`, posts via `gh api .../comments` or `.../reviews`.
- **State changes on PRs:** `gh pr merge` (any flags), `gh pr close`, `gh pr reopen`, `gh pr ready` (and `--undo`), `gh pr edit`.
- **CI:** `gh run rerun`, `gh run cancel`.
- **Issues:** `gh issue close`, `gh issue reopen`, `gh issue edit`, `gh issue delete`.
- **Releases:** `gh release create`, `gh release edit`, `gh release delete`.
- Any `gh api -X POST/PATCH/PUT/DELETE` that mutates state, including resolving review threads.
- **Git remote ops:** pushing branches, force-push, deleting branches/tags.

Read-only commands (`list`, `view`, `diff`, `checks`, `status`, `gh api` GETs) are fine. When in doubt, surface the command and wait.

### Output Discipline

- `gh run view --log` is huge — prefer `--log-failed` or pipe through `tail -N`.
- `gh api ... --paginate` can be massive — add `--jq` to filter.
- `gh pr diff` on big PRs — use `--name-only` first, then targeted reads.

---

## 8. Violations

Any deviation from these rules is a **blocker**. The responsible agent must stop, revert the offending change, and correct the violation before continuing work.
