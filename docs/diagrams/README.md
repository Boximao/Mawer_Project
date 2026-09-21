# Diagrams

Two presentation diagrams for the workshop demonstration. The `.svg` files are the source; the
`.png` files are 2x renders for slides, the TV and anything that cannot display SVG.

| File | Audience | What it shows |
|---|---|---|
| `product-design.svg` / `.png` | Non-technical reviewers | Who the assistant is for, the five-stage journey (ask, ground, draft with evidence, check, human release), the only two possible outcomes, and the design principles behind them. |
| `technical-architecture.svg` / `.png` | Engineers | Repository layout, the offline ingest path, the three runtime planes (talk, control, data), the Neon schema and guards, and the run state machine. |
| `agentic-workflow.svg` / `.png` | Mixed room | One run from question to answer, what the run records for auditing (`runs`, `agent_steps`, `approvals`), and what the offline evaluation scores. |

Both stay consistent with [`../AGENTIC_RAG_SPEC.md`](../AGENTIC_RAG_SPEC.md) (the contract),
[`../TECHNICAL_SOLUTION.md`](../TECHNICAL_SOLUTION.md) and
[`../RUNNING_ENVIRONMENT.md`](../RUNNING_ENVIRONMENT.md). If the spec changes, edit the `.svg` and
re-render.

## Re-rendering the PNGs

```powershell
powershell -ExecutionPolicy Bypass -File docs\diagrams\render.ps1
```

The script inlines each SVG into a zero-margin HTML page and screenshots it with headless Chrome
(falling back to Edge) at twice the declared size. Keep the SVG text ASCII-only and write special
glyphs as XML entities (`&#8212;` for an em dash, `&#183;` for a middle dot); literal non-ASCII
characters do not survive the editing path and render as empty boxes.
