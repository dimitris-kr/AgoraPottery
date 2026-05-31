@echo off
setlocal
REM ──────────────────────────────────────────────────────────────
REM Deploy 03_web_app/ to the Hugging Face Space.
REM
REM Pushes a SQUASHED single commit of the current backend files to the `hf-space` remote's main branch.
REM Single commit = no git history is sent, old commits ignored = can't be rejected by HF.
REM GitHub `main` is never touched.
REM
REM Run from the repo root, with a clean working tree (commit + push to GitHub first).
REM deploy_web_app.bat
REM ──────────────────────────────────────────────────────────────

echo === Deploying 03_web_app to the HF Space ===

REM Remove any leftover temp branches from a previous run
git branch -D hf-deploy >nul 2>&1
git branch -D hf-clean  >nul 2>&1

REM 1. Create temp branch with 03_web_app contents at root
git subtree split --prefix 03_web_app -b hf-deploy || goto :fail

REM 2. Re-create it to second temp branch as a single orphan commit - no history
git checkout --orphan hf-clean hf-deploy || goto :fail
git commit -m "Deploy 'Pottery Chronology Predictor' API" || goto :fail

REM 3. Force-push the single-commit clean branch to the Space main branch
git push hf-space hf-clean:main --force || goto :fail

REM 4. Back to main, clean up temp branches
git checkout main
git branch -D hf-deploy hf-clean >nul 2>&1
echo === Deploy complete ===
goto :eof

:fail
echo === Deploy FAILED ===
git checkout main
git branch -D hf-deploy hf-clean >nul 2>&1
exit /b 1
