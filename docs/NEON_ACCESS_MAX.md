# Neon access for Max

Pull this file from the repo. It is the checklist to open the shared Lakebase Postgres (Neon) project and put a connection string in local `.env`.

**Do not commit `.env` or any password.** Jason and Max share one pooled `DATABASE_URL` locally; the password lives only in the Neon Console and in gitignored `.env`.

---

## Your access (already granted)

| Item | Value |
|---|---|
| Email | `a0917664896@gmail.com` |
| Organization role | **Editor** (connection strings + SQL on every project in the org) |
| Member since | 2026-09-21T20:16:34Z |
| Sign-in | [https://console.neon.tech](https://console.neon.tech) with this Gmail |

If the Console asks you to join an organization, accept **YupoOrg**. The pending invite was consumed when this account joined.

---

## Direct URLs

Open these after you are signed in:

| What | URL |
|---|---|
| Neon Console (sign in) | https://console.neon.tech |
| Organization | https://console.neon.tech/app/org-polished-fog-91921767 |
| **Project dashboard (start here)** | https://console.neon.tech/app/projects/dawn-moon-44249230 |
| SQL Editor (production branch) | https://console.neon.tech/app/projects/dawn-moon-44249230/branches/br-fancy-flower-b5dfjw7e/sql-editor |

On the project dashboard, click **Connect**. That modal is the source of the live password and the copy-paste `DATABASE_URL`.

---

## Project identifiers

| Item | Value |
|---|---|
| Organization | YupoOrg |
| Org ID | `org-polished-fog-91921767` |
| Project name | MIM_Trial |
| Project ID | `dawn-moon-44249230` |
| Region | `aws-us-east-2` (Ohio) |
| Postgres | 18 |
| Default / production branch | `production` |
| Branch ID | `br-fancy-flower-b5dfjw7e` |
| Database | `neondb` |
| Role | `neondb_owner` |
| Compute ID | `ep-odd-mountain-b5g40xpv` |
| Port | `5432` |
| SSL | required (`sslmode=require`) |

---

## Hosts (no password in git)

Use **pooled** for the app. Use **direct** (no `-pooler`) for schema migrations (`scripts/init_cloud_db.py`).

| Kind | Host |
|---|---|
| Direct | `ep-odd-mountain-b5g40xpv.c-7.us-east-2.aws.neon.tech` |
| Pooled | `ep-odd-mountain-b5g40xpv-pooler.c-7.us-east-2.aws.neon.tech` |

URI shape (replace `PASSWORD` from **Connect** in the Console):

```
postgresql://neondb_owner:PASSWORD@ep-odd-mountain-b5g40xpv-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require
```

Direct (migrations):

```
postgresql://neondb_owner:PASSWORD@ep-odd-mountain-b5g40xpv.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require
```

---

## Local `.env` (repo root)

```powershell
copy .env.example .env
```

Set at least:

```
DATABASE_URL=postgresql://neondb_owner:PASSWORD@ep-odd-mountain-b5g40xpv-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require
DATABASE_URL_UNPOOLED=postgresql://neondb_owner:PASSWORD@ep-odd-mountain-b5g40xpv.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require
```

Paste the pooled string from the Console **Connect** widget into `DATABASE_URL`. Turn connection pooling **off** in that widget to copy `DATABASE_URL_UNPOOLED`.

Wake compute with `SELECT 1` in the SQL Editor (or any client) if the first query after idle is slow.

---

## Apply schema from this repo

From the repo root (direct URL in `.env`):

```powershell
.\.venv\Scripts\python.exe scripts\init_cloud_db.py
```

SQL lives in `migrations/001_initial.sql`. Shared store: pgvector + HITL tables on this same database. Runtime notes: [`RUNNING_ENVIRONMENT.md`](./RUNNING_ENVIRONMENT.md).
