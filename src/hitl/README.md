# HITL (Jason)

HMAC-SHA256 + Postgres trigger. Spec §6.

Server signs after PIN check (`HITL_SIGNING_KEY` never in the prompt or client). Demo PIN in `demo/app.js` is talk-only when the UI is opened as a file.
