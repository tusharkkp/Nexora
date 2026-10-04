# Skills for the Code Review Agent

Modular skills the agent invokes during review. Each skill defines **when to use it**, **how**, and **what it outputs**. Tool output is always a *lead*, never a conclusion: verify against the code before reporting.

> **Safety rule for all skills:** run analysis tools in read-only mode. Do not run install scripts, builds, or tests from an untrusted repo outside a sandbox.

---

## Skill 1: `repo-recon`

**Use when:** Always, as the first step.

**Procedure**
```bash
git clone --depth 1 <repo_url> repo && cd repo      # add --branch <ref> if specified
git log -n 1 --format='%H %an %ad %s'               # record the reviewed commit SHA
git rev-list --count HEAD                           # (use full clone if history matters)
tokei . || cloc .                                   # language + LOC breakdown
find . -maxdepth 2 -type f \( -iname 'README*' -o -iname 'LICENSE*' -o -iname 'SECURITY*' -o -iname 'CONTRIBUTING*' \)
ls -a                                               # look for CI, Docker, lockfiles, configs
```
Look for: `.github/workflows/`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `package.json`, `pyproject.toml`, `requirements*.txt`, `go.mod`, `Cargo.toml`, `pom.xml`, `.env.example`.

**Output:** project summary, tech stack, entry points, size, excluded paths (vendored/generated), maturity estimate.

---

## Skill 2: `architecture-mapping`

**Use when:** After recon, for `standard` and `deep` reviews.

**Procedure**
1. List top-level directories and state each one's responsibility.
2. Find entry points (`main`, `app`, `server`, `cli`, route definitions, job schedulers).
3. Trace 1-3 critical flows from entry to persistence/output.
4. Build an import graph for key modules; look for cycles and modules that everything depends on.
   ```bash
   # Python (example)
   pydeps <package> --noshow --max-bacon=2   # if available
   # JS/TS (example)
   npx madge --circular --extensions ts,js src/
   ```
5. Identify where state lives (DB, cache, globals, files) and who owns it.

**Output:** component list, a Mermaid diagram, and architecture findings (coupling, layering, cycles, god modules).

---

## Skill 3: `secrets-scan`

**Use when:** Always.

**Procedure**
```bash
gitleaks detect --source . --no-banner --redact          # includes git history
trufflehog filesystem . --only-verified=false            # alternative
grep -rIn --exclude-dir=.git -E "(api[_-]?key|secret|passwd|password|token|private[_-]?key)\s*[:=]" . | head -100
git log --all --diff-filter=D --name-only | grep -Ei '\.env|\.pem|id_rsa|credentials'   # deleted sensitive files
```
Check `.gitignore` covers `.env`, keys, and local config.

**Rules:** Redact values in output. Recommend **rotation**, since deletion doesn't remove git history. Distinguish real secrets from placeholders and test fixtures.

**Output:** list of location + secret type + exposure (current tree vs. history) + severity.

---

## Skill 4: `dependency-audit`

**Use when:** A manifest or lockfile exists.

**Procedure**
```bash
# Cross-ecosystem
osv-scanner --lockfile=<lockfile>            # or: osv-scanner -r .

# Python
pip-audit -r requirements.txt
pip list --outdated                          # only in a sandbox

# Node
npm audit --omit=dev --package-lock-only
npx depcheck                                 # unused deps

# Go / Rust
govulncheck ./...
cargo audit
```
Also check: lockfile committed? Versions pinned or floating (`^`, `*`, `latest`)? Abandoned packages (no release in 2+ years)? License compatibility (`license-checker`, `pip-licenses`)? Suspicious install scripts or typosquat-looking names?

**Output:** vulnerable deps with CVE/GHSA ID, whether the vulnerable code path is actually reachable, upgrade path, and unused/abandoned/licensing issues.

---

## Skill 5: `static-analysis`

**Use when:** `standard` or `deep` depth, or `focus` includes security/correctness.

**Procedure** (use whatever matches the stack)
```bash
semgrep --config auto --metrics=off .                    # multi-language security/correctness
bandit -r . -x tests                                     # Python security
ruff check . && mypy .                                   # Python lint + types
npx eslint . && npx tsc --noEmit                         # JS/TS
golangci-lint run                                        # Go
cargo clippy -- -D warnings                              # Rust
hadolint Dockerfile                                      # Dockerfile
checkov -d . --quiet                                     # IaC (Terraform, K8s, CloudFormation)
actionlint                                               # GitHub Actions workflows
```

**Rules:** Triage every result. Open the file, confirm it's real and reachable, and drop false positives. Group repeated findings by root cause.

**Output:** verified findings only, mapped to the finding format in `agent.md`.

---

## Skill 6: `security-deep-dive`

**Use when:** `focus` includes security, or the project handles auth, payments, PII, or untrusted input.

**Procedure**
1. Enumerate **trust boundaries**: HTTP endpoints, CLI args, file uploads, queues, webhooks, DB reads of user-supplied data, LLM/tool outputs.
2. For each input, trace to **sinks**: SQL, shell, filesystem, templates, deserializers, HTTP clients, `eval`/`exec`, redirects.
3. For each endpoint, verify authentication **and** object-level authorization (can user A access user B's resource by changing an ID?).
4. Review session/JWT handling, password storage (bcrypt/argon2/scrypt vs. fast hashes), token expiry and rotation.
5. Review CORS, CSRF, cookie flags (`HttpOnly`, `Secure`, `SameSite`), security headers, rate limiting.
6. Review CI/CD: workflow permissions, `pull_request_target`, third-party actions pinned to SHAs, secrets exposure to forks.
7. Review Dockerfile/IaC: non-root user, pinned base image, exposed ports, privileged mode, wildcard IAM.

**Output:** security findings with an attack scenario for each Critical/High.

---

## Skill 7: `test-and-quality-analysis`

**Use when:** Always at `standard`+; `deep` may run tests in a sandbox.

**Procedure**
1. Locate tests; map them to modules. Note modules with none.
2. Read a sample of tests: do they assert behavior, or just execute code? Over-mocked?
3. Look for flakiness patterns: `sleep`, wall-clock time, randomness without seeds, shared global state, order dependence.
4. Inspect CI: does it run tests, lint, types, and security scans on PRs? Branch protection signals?
5. *(Sandbox only, if allowed)* Run the suite and coverage:
   ```bash
   pytest --cov --cov-report=term-missing -q
   npm test -- --coverage
   ```
6. Prioritize coverage gaps by risk (auth, money, data mutation), not by percentage.

**Output:** test inventory, gaps on critical paths, flakiness risks, CI gaps, and a practical "tests to add first" list.

---

## Skill 8: `performance-review`

**Use when:** `focus` includes performance, or the code serves traffic or processes large data.

**Procedure**
1. Find hot paths: request handlers, loops over collections, data pipelines, startup code.
2. Look for N+1 queries (queries inside loops, ORM lazy loading), unbounded `SELECT`, missing pagination, missing indexes (if schema/migrations visible).
3. Look for blocking I/O in async code, synchronous network calls in request paths, missing timeouts.
4. Check complexity of algorithms on large inputs; repeated work that could be cached or batched.
5. Check memory behavior: loading whole files/datasets, unbounded caches/queues, large object retention.
6. Check connection pooling, retry/backoff, and concurrency limits.

**Rule:** Only report issues you can justify from the code. Label estimated impact as an estimate.

**Output:** performance findings with the scenario in which they hurt (e.g., "at 10k rows this issues 10k queries").

---

## Skill 9: `ai-ml-review`

**Use when:** Repo contains ML training, data pipelines, LLM calls, RAG, or agents.

**Procedure**
1. **Data pipeline:** Is the train/val/test split done before preprocessing/fitting (scalers, encoders, feature selection)? Any target/label leakage? Time-series split respected?
2. **Reproducibility:** Seeds set (Python, NumPy, framework)? Dependencies and model versions pinned? Config and hyperparameters externalized? Experiment tracking?
3. **Evaluation:** Metrics fit the task and class balance? Baseline present? Held-out evaluation, not training data? Error analysis?
4. **Serving:** Model loaded once, not per request? Input validation, timeouts, batching, fallback behavior?
5. **LLM/agent safety:**
   - Does untrusted text (web pages, emails, documents, user input) flow into prompts or tool arguments? (prompt injection)
   - Are tool permissions least-privilege? Are tool arguments validated?
   - Is output validated against a schema before use? Any `eval`/`exec` of model output?
   - Are there timeouts, retries with backoff, token/cost caps, and rate limits?
   - Are secrets or PII sent to third-party APIs or logged?
   - Is there any evaluation harness or regression test for prompts?
6. **RAG:** Chunking and overlap rationale, embedding model/version consistency between indexing and querying, retrieval evaluation, handling of stale or deleted documents, whether answers cite sources.
7. **Unsafe deserialization:** `pickle.load`, `torch.load` without `weights_only=True`, `joblib.load`, or `yaml.load` on untrusted files.

**Output:** AI/ML findings in the standard format, plus a short "evaluation and reproducibility" assessment.

---

## Skill 10: `docs-and-dx-review`

**Use when:** Always (lightweight at `quick`).

**Procedure**
1. Follow the README setup steps mentally (or in a sandbox): are prerequisites, env vars, and commands complete and in order?
2. Check for `.env.example`, config documentation, architecture notes, API docs, contribution guide, license.
3. Compare docs to code: stale commands, renamed flags, nonexistent files.
4. Estimate onboarding friction: steps to first run, undocumented assumptions.

**Output:** docs findings and a short "time to first run" assessment.

---

## Skill 11: `finding-triage-and-report`

**Use when:** Always, as the final step.

**Procedure**
1. Collect all candidate findings from every skill.
2. **Verify:** reopen each cited location; confirm the line numbers and behavior.
3. **Dedupe:** merge findings sharing a root cause (e.g., 14 unvalidated inputs → one "no input validation layer" finding with a list of locations).
4. **Rate:** assign severity (impact × likelihood in this project's context) and confidence.
5. **Prioritize:** select the top 3-5 actions by leverage.
6. **Redact:** remove secret values and personal data.
7. **Render:** write the report using the structure in `agent.md` Section 8.
8. **Self-check:** run the checklist in `agent.md` Section 10.

**Output:** the final report.

---

## Skill Selection Guide

| Depth | Skills run |
|---|---|
| `quick` | 1, 3, 10, light 5, 11 (report Critical/High only) |
| `standard` | 1, 2, 3, 4, 5, 7, 10, 11 + any focus-specific skills (6, 8, 9) |
| `deep` | All skills, plus sandboxed test/coverage runs if permitted |

| Focus | Add |
|---|---|
| `security` | 6 (and prioritize 3, 4) |
| `performance` | 8 |
| `architecture` | 2 (expanded) |
| `testing` | 7 (expanded) |
| `ai-ml` | 9 |
