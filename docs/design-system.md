# devin-local design system

This document is the Figma-equivalent spec for the devin-local GUI. Instead of a `.fig`
file (which would require live Figma access we don't have for a fully local app),
the design lives in three places that stay in sync:

1. **Tokens** — `src/devin_local/gui/design_tokens.py`. Single source of truth for
   colors, spacing, type, radii, motion.
2. **Stylesheet** — `src/devin_local/gui/theme.py`. QSS rendered from the same
   palette, applied to the whole `QApplication`.
3. **Components** — widgets under `src/devin_local/gui/widgets.py`,
   `sandbox_panel.py`, `model_selector.py`, `settings_dialog.py`. Each is documented
   below with its states and the tokens it consumes.

When you change a token in `design_tokens.py`, mirror the change here.

---

## 1. Color tokens (tokyo-night)

| Token | Hex | Use |
| --- | --- | --- |
| `bg.root` | `#0d0f17` | App background |
| `bg.chrome` | `#13151f` | Sidebar, inspector, status bar |
| `bg.panel` | `#1a1b26` | Cards (tool, plan, sandbox), bubbles |
| `bg.panel-alt` | `#24283b` | Composer, hover surface |
| `bg.input` | `#13151f` | Text inputs |
| `border` | `#2f3549` | All hairline rules |
| `border.strong` | `#3b4261` | Focus rings, dropdown menus |
| `text` | `#c0caf5` | Primary text |
| `text.muted` | `#9aa5ce` | Secondary text |
| `text.subtle` | `#565f89` | Disabled, hints |
| `accent` | `#7aa2f7` | Primary buttons, links, selected tab |
| `accent.hover` | `#bb9af7` | Hover state on accent |
| `accent.press` | `#5e87df` | Pressed state on accent |
| `success` | `#9ece6a` | OK pills, "ready" badge, completed plan step |
| `warn` | `#e0af68` | Warnings, "missing dep" badge |
| `error` | `#f7768e` | Error pills, failed plan step |
| `info` | `#7dcfff` | Informational pills, links in chat |
| `code.bg` | `#11141b` | Code-block background |

**Contrast**: `text` on `bg.root` measures ≥ 11:1 (WCAG AAA for normal text).
`text.muted` on `bg.root` is ≥ 6:1 (AAA for large text, AA for normal).

---

## 2. Spacing

A 4px base scale. Compose paddings/margins from these tokens — never raw px.

| Token | px | Typical use |
| --- | --- | --- |
| `space.xxs` | 2 | Pill internal padding |
| `space.xs` | 4 | Tight gaps (icon ↔ text) |
| `space.sm` | 8 | Default gap between siblings |
| `space.md` | 12 | Card internal padding |
| `space.lg` | 16 | Section spacing |
| `space.xl` | 24 | Pane padding (sidebar, inspector) |
| `space.xxl` | 32 | Hero sections |

---

## 3. Radii

| Token | px | Use |
| --- | --- | --- |
| `radius.sm` | 4 | Inputs, secondary buttons |
| `radius.md` | 8 | Cards, primary buttons |
| `radius.lg` | 12 | Bubbles, dialogs |
| `radius.pill` | 999 | Status pills, badges |

---

## 4. Typography

| Token | Value |
| --- | --- |
| `font.ui` | `Inter, Segoe UI, SF Pro Text, system-ui, sans-serif` |
| `font.mono` | `JetBrains Mono, Cascadia Mono, Consolas, monospace` |
| `size.xs` | 11 px |
| `size.sm` | 12 px |
| `size.md` | 13 px (body default) |
| `size.lg` | 15 px |
| `size.xl` | 18 px (section headings) |
| `size.xxl` | 22 px (page title) |
| `weight.regular` | 400 |
| `weight.medium` | 500 |
| `weight.bold` | 600 |

---

## 5. Motion

| Token | ms | Use |
| --- | --- | --- |
| `motion.fast` | 120 | Hover state transitions |
| `motion.medium` | 200 | Card open/close, pill state changes |
| `motion.slow` | 320 | Dialog enter/exit |

Easing: default to `ease-out` for entrances, `ease-in` for exits, `ease-in-out` for
state changes.

---

## 6. Components

### 6.1 Sidebar

- Background: `bg.chrome`, right border: `border`.
- Width: 220–320px, resizable via `QSplitter`.
- Items:
  - **Title** ("devin-local") in `size.xxl`, `weight.bold`.
  - **Version** in `size.sm`, `text.subtle`.
  - **+ New session** primary button (full width, `accent` bg).
  - **Sessions** list (`QListWidget`).
  - **Settings** button (full width, secondary). Opens `SettingsDialog`.

### 6.2 Chat pane

- Background: `bg.root`.
- Vertical layout: `ChatPane` (scroll area of bubbles) on top, `Composer` on bottom.
- Bubbles:
  - **User bubble**: `bg.panel-alt` background, `text` text, right-aligned, `radius.lg`,
    `space.md` padding. Max width 80% of chat pane.
  - **Assistant bubble**: `bg.panel` background, `text` text, left-aligned, with a
    leading `◆` glyph in `accent`. Code fences are extracted into separate
    `CodeBlockWidget` instances (see 6.5).
  - **System notice**: centered, `text.subtle`, italic.

### 6.3 Composer

- Background: `bg.panel-alt`, top border: `border`.
- Multi-line `QPlainTextEdit` with placeholder "Send a message…".
- **Send** button: `accent` bg, white text, `radius.md`. Disabled while the agent is
  running.
- Keyboard: `Ctrl+Enter` submits, `Esc` clears.

### 6.4 Tool card

- Layout: header row + collapsible args panel + collapsible result panel.
- Header:
  - **Icon** (from `icons.py`): `tool` glyph in `accent`, 14×14.
  - **Name** (e.g. `write_file`): `size.md`, `weight.medium`.
  - **Summary** (first arg or short description): `size.sm`, `text.muted`.
  - **Status pill**:
    - `running`: `info` bg-10%, `info` border, `info` text, content "running".
    - `ok`: `success` bg-10%, `success` border, `success` text, content "ok 12ms".
    - `error`: `error` bg-10%, `error` border, `error` text, content "error".
  - **Elapsed**: monospace `text.subtle`, right-aligned.
- Background: `bg.panel`, `radius.md`, `space.md` padding.
- File-output card variant (`write_file`/`edit_file`): result panel shows a
  language-highlighted preview of the file pulled from disk.

### 6.5 Code block

- Background: `code.bg`, border: `border`, `radius.md`.
- Header row: language label (`size.sm`, `text.muted`), copy button (top-right).
- Body: Pygments-highlighted text in `font.mono`. Falls back to plain monospace
  when Pygments isn't installed.

### 6.6 Plan pane

- Lives in the right inspector under `PLAN`.
- Each step is a row with a leading glyph reflecting its status:
  - `pending` → `○` in `text.subtle`
  - `in_progress` → `◐` in `accent`
  - `completed` → `✔` in `success`
  - `failed` → `✖` in `error`
- Summary line below: `4/7 done · 1 in progress`.

### 6.7 Sandbox panel

- Lives in the right inspector under `SANDBOX`.
- Sections (each a small card with `bg.panel`, `radius.md`):
  - **Workspace**: path + size + "open in file manager" link.
  - **Terminal**: cwd, last command (mono), exit code pill.
  - **Desktop**: emulator state (idle / typing / clicking) + screenshot thumbnail.
  - **Tools**: chips listing every enabled tool name.
  - **Flags**: chips for `net`, `browser`, `desktop`, `mcp` with on/off state.

### 6.8 Settings dialog

- `QDialog` with `QTabWidget`. Tabs in this order: **General**, **MCP**, **GitHub**,
  **Backends**, **Appearance**.
- Each tab uses a `QFormLayout` for label/control pairs.
- **General**: workspace (line edit + Browse), default model, default backend,
  Ollama host, parallel-tool-calls, OBLITERATUS, plan-first toggles.
- **MCP**: list widget + Add / Edit / Remove / Test connection buttons. Editor
  dialog has name, transport (stdio/http/sse), command, args, URL, enabled.
- **GitHub**: username (read-only after Test), PAT (password masked), Test
  connection button. PAT saved to `~/.devin-local/github.json` with `0600` perms.
- **Backends**: row per backend (`ollama`, `layered`, `hf`) with current probe
  status and an Install button that opens the existing `InstallBackendDialog`.
- **Appearance**: theme (currently only `tokyo-night`), font scale (0.75–1.75),
  accent hex, show-sandbox + show-plan toggles.
- Footer: `Save` / `Cancel`. Save persists all tabs, fires `settings_saved` signal.

### 6.9 Status bar

- Background: `bg.chrome`, top border: `border`.
- Left → right: state pill (idle/streaming/tool), backend, model, workspace path.
- All text in `size.sm`, `text.muted`.

---

## 7. Icons

Defined in `src/devin_local/gui/icons.py`. Logical name → (Material/qtawesome glyph,
Unicode fallback). Always reach via `icon("name")` so the icon family is swappable
in one place.

Set: `send`, `settings`, `plus`, `minus`, `trash`, `refresh`, `github`, `robot`,
`user`, `play`, `check`, `x`, `warn`, `spark`, `folder`, `file`, `terminal`,
`sandbox`, `plan`, `copy`, `search`, `link`, `tool`, `model`, `backend`, `mcp`,
`appearance`, `general`.

`qtawesome` is in the `[gui]` extra; if it's missing, every icon falls back to
its Unicode glyph.

---

## 8. Accessibility

- Minimum tap target: 32×32 px.
- All interactive elements have focus rings (`border.strong`, 2px).
- Status pills additionally encode state via leading glyph (color is never
  the only signal).
- Keyboard: every dialog can be navigated with Tab / Shift+Tab; Enter activates
  the default button; Esc cancels.

---

## 9. How to extend

Adding a component? Write it under `src/devin_local/gui/`, consume tokens via
`from devin_local.gui.design_tokens import TOKENS`, and add a row in section 6
above. Adding a token? Edit `design_tokens.py` and mirror it in section 1–5.
