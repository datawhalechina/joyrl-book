---
name: interactive-chapter
description: Plan and implement interactive chapter or playground pages for this Docusaurus repo. Use when Codex needs to port AI-generated JSX into JoyRL Book, add a standalone interactive route under `src/pages/`, link a chapter under `docs/` to that route, or decide whether missing infrastructure such as Tailwind, shared navigation, or TOC injection must be added first.
---

# Interactive Chapter

## Overview

Port the original CS123 interactive-chapter workflow into this repo without assuming the source repo's file layout exists here. Favor the smallest working integration first: place React code deliberately, wire a normal Docusaurus page, link it from the chapter docs, and validate the site build.

## Repo Snapshot

- Docs live under `docs/`; many chapters are `docs/rl_basic/ch*/README.md`.
- Standalone pages can live under `src/pages/`.
- Global styling lives in `src/css/custom.css`.
- This repo currently does not include a checked-in `src/components/` tree, Tailwind, a `PlaygroundHeader`, a `PLAYGROUNDS` registry, or a TOC swizzle that injects interactive links.

Treat those missing pieces as explicit product decisions, not safe assumptions.

## Workflow

### 1. Confirm the shape of the interactive experience

- Prefer a normal docs update when the request is only a small inline visualization or explanation.
- Prefer a standalone page under `src/pages/` when the UI is full-screen, stateful, or demo-heavy.
- Avoid inventing a shared navigation system unless the user asks for a reusable pattern across multiple chapters.

### 2. Normalize incoming React or JSX

- Treat AI-generated JSX as source material, not as drop-in code.
- Rewrite styling for the current repo instead of assuming Tailwind or a `tw-` class prefix exists.
- Translate Tailwind-heavy output into plain CSS, CSS modules, or carefully scoped rules in `src/css/custom.css` unless the user explicitly wants Tailwind added as a separate task.
- Remove source-repo assumptions such as `PlaygroundHeader`, `PLAYGROUNDS`, or automatic TOC button injection.

### 3. Place files deliberately

- Create reusable UI under `src/components/interactive/<slug>/` when the same widget will appear in more than one place.
- Create a page wrapper under `src/pages/` for the user-facing route.
- Link to that route from the relevant Markdown chapter in `docs/` with a normal Markdown link or callout.
- Keep naming consistent between the docs slug, component folder, and page route.

Example page wrapper:

```jsx
import React from 'react';
import Layout from '@theme/Layout';
import Demo from '@site/src/components/interactive/<slug>/Demo';

export default function DemoPage() {
  return (
    <Layout title="Interactive Demo" description="Interactive chapter demo">
      <Demo />
    </Layout>
  );
}
```

### 4. Keep chapter integration simple

- Update the relevant chapter file under `docs/` with an explicit link to the interactive page.
- Do not assume the chapter must become MDX unless importing React directly into the document is necessary.
- Do not edit global navigation or sidebar structure unless the new route must be surfaced there.

### 5. Validate before finishing

- Run `npm run typecheck` when the change introduces or touches typed config.
- Run `npm run build` after adding pages, links, or assets.
- Check that the generated route and the source chapter both build cleanly.

## Translation Notes From the Source Skill

This skill was migrated from `/Users/johnjim/Desktop/dive-into-embodied-ai/.claude/skills/interactive-chapter/SKILL.md`.

The original version assumed all of the following:

- `docs/projects/cs123/...` chapter paths
- `src/components/PlaygroundHeader/index.jsx`
- a `PLAYGROUNDS` array that drives previous and next navigation
- `src/theme/DocItem/TOC/Desktop/index.tsx` customization
- Tailwind configured with `prefix: 'tw-'`

None of those assumptions are present in `joyrl-book` today. Port them only when the user explicitly asks for the full CS123-style experience.

## Guardrails

- Do not add Tailwind or any other new dependency silently.
- Do not create a shared interactive framework for a one-off demo unless reuse is clearly likely.
- Do not preserve source-repo file paths in copied code without checking that the targets exist here.
- Do not claim automatic TOC or previous/next behavior unless that infrastructure has actually been implemented.
