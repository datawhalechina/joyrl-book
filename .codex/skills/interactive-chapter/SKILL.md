---
name: interactive-chapter
description: Plan and implement interactive chapter or playground pages for this Docusaurus repo. Use when Codex needs to port AI-generated JSX into JoyRL Book, add a standalone interactive route under `src/pages/`, wire a chapter into interactive mode, or extend the current left-reading/right-playground interaction pattern.
---

# Interactive Chapter

## Overview

Build interactive chapters in `joyrl-book` using the repo's current integration pattern instead of re-inventing chapter routing, TOC entry points, or page layout each time.

Favor the existing chapter workflow first:

- docs chapter stays under `docs/...`
- interactive UI lives under `src/components/interactive/<slug>/`
- route page lives under `src/pages/.../playground.jsx`
- chapter-to-interactive mapping lives in `src/components/interactive/interactiveRoutes.ts`

## Current Repo Pattern

These pieces already exist in this repo and should be reused unless the user explicitly asks to replace them:

- `src/components/interactive/interactiveRoutes.ts`
  Maps docs to interactive routes, exposes reading-mode helpers, and stores RL chapter order for chapter switching.
- `src/theme/DocItem/index.tsx`
  Redirects supported docs into interactive mode on the client unless `?view=read` is present.
- `src/theme/DocItem/TOC/Desktop/index.tsx`
  Injects the desktop right-rail `交互模式` button above the TOC.

Treat those as real infrastructure, not hypothetical future work.

## Default UX To Preserve

For full interactive chapters, prefer this desktop pattern unless the user asks for a different design:

- top toolbar for `阅读模式`, chapter switcher, previous chapter, next chapter
- left column for reading-mode-style explanation and controls
- right column for interactive panels only
- page-level previous/next navigation inside the left column
- avoid duplicating explanatory copy on both left and right

For control-heavy pages such as a console page:

- keep the explanation on the left
- keep formulas, charts, or playground panels on the right
- use a two-column control grid on desktop when the controls are numerous

## Workflow

### 1. Confirm the scope

- Prefer a normal docs edit for a small inline visualization.
- Prefer the existing interactive-chapter pattern for full-screen, stateful, multi-step teaching UIs.
- Reuse the current repo pattern before inventing a new shared framework.

### 2. Normalize incoming React or JSX

- Treat AI-generated JSX as source material, not drop-in code.
- Rewrite styling into CSS modules or existing repo CSS patterns.
- Do not assume Tailwind, `tw-` classes, `PlaygroundHeader`, or source-repo globals exist.

### 3. Place files deliberately

- Put reusable chapter UI in `src/components/interactive/<slug>/`
- Put the user-facing route in `src/pages/<doc path>/playground.jsx`
- Keep names aligned between doc slug, component folder, and route

Typical page wrapper:

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

### 4. Register the chapter

When a docs page should support interactive mode:

1. Add an entry to `INTERACTIVE_DOC_MAP` in `src/components/interactive/interactiveRoutes.ts`
2. Reuse `getReadingModeHref(...)` for the reading-mode link
3. If the chapter participates in the RL chapter switcher, keep `RL_BASIC_CHAPTERS` accurate

Do not add a redundant inline Markdown link by default when the chapter already has:

- TOC `交互模式` button
- default redirect into interactive mode

Only add explicit in-doc links when the user wants visible duplication.

### 5. Preserve the reading/interactive split

When refining an interactive page, prefer these heuristics:

- left side owns narrative copy, bullets, and operation hints
- right side owns metrics and visual/interactive state
- if the same title already appears in the page-level pager, remove extra duplicates
- if the same explanation appears on both sides, keep it on the left

### 6. Validate before finishing

- Run `npm run typecheck` when changing typed config or shared routing helpers
- Run `npm run build` after adding pages, links, or layout integration
- Confirm the docs route and interactive route both still build

## Current File Landmarks

- interactive route registry:
  `src/components/interactive/interactiveRoutes.ts`
- doc redirect:
  `src/theme/DocItem/index.tsx`
- desktop TOC button:
  `src/theme/DocItem/TOC/Desktop/index.tsx`
- current reference implementation:
  `src/components/interactive/dqn/DqnPlayground.jsx`

## Guardrails

- Do not add Tailwind or another styling system silently.
- Do not claim automatic TOC button, redirect, or chapter switching unless they are actually wired.
- Do not preserve source-repo file paths without checking they exist here.
- Do not create a second routing pattern when the current `interactiveRoutes.ts` setup already solves the task.
