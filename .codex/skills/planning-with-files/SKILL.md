---
name: planning-with-files
description: File-based planning workflow using `task_plan.md`, `findings.md`, and `progress.md` to track multi-phase tasks across many tool calls.
---

# planning-with-files

File-based planning for complex tasks (persistent “working memory on disk”).

Use this when a task will take many tool calls, spans multiple phases, or benefits from explicit tracking (e.g., research + implementation + verification).

## Where files go

- Skill assets (templates/scripts): `./.codex/skills/planning-with-files/`
- Per-task planning files (created in the project root):
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## Quick start

```bash
./.codex/skills/planning-with-files/scripts/init-session.sh pto-wsp
```

## Workflow rules

- Create `task_plan.md` before doing complex work.
- Re-read `task_plan.md` before major decisions (keeps goals “in attention”).
- After discoveries (especially from browsing/large outputs), write key points into `findings.md`.
- Log errors and resolutions (at minimum in `task_plan.md`; optionally detailed in `progress.md`).
- Don’t repeat the same failing action; change approach.

## Helpers

- Initialize planning files: `./.codex/skills/planning-with-files/scripts/init-session.sh`
- Completion gate (optional): `./.codex/skills/planning-with-files/scripts/check-complete.sh [task_plan.md]`

## References

- Manus context-engineering notes: `./.codex/skills/planning-with-files/reference.md`
- Worked examples: `./.codex/skills/planning-with-files/examples.md`
