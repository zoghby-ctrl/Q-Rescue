# Q-Rescue Streamlit Community Cloud Deployment Guide

The repository is prepared for deployment, but this guide does not authorize a public GitHub push or a public Streamlit deployment.

## Deployment profile

- Entrypoint: `app.py` at repository root.
- Working directory: repository root.
- Python: **3.12**, selected in Streamlit Community Cloud Advanced settings.
- Dependency file: root `requirements.txt` (runtime only).
- Configuration: `.streamlit/config.toml`.
- Secrets: none for the default demo; do not upload a secrets file.
- Default compute: synthetic grid, rain, K=4, feasible-XY, p=1.
- Advanced compute: K=8 and p=3 are supported but materially heavier; they are not the default presentation scenario.

Streamlit Community Cloud copies the repository, runs from its root, and installs the root dependency file. The app resolves all runtime data from `Path(__file__)`, so it does not depend on the uploader's current directory or an absolute local path.

## 1. Prepare GitHub

1. Resolve the license, citation-author, manuscript-privacy, and archive-boundary items in `GITHUB_RELEASE_GUIDE.md`.
2. Initialize and review Git locally if this is the first release:

   ```powershell
   git init -b main
   git add .
   git status --short
   git diff --cached --stat
   ```

3. Confirm that the exact MVP deployment files listed in `GITHUB_RELEASE_GUIDE.md` are staged, especially the model `.joblib` files, calibration files, research-summary CSVs, cached `.graphml` files, and `paper/CLAIMS.md`.
4. Commit locally, create a GitHub repository, add the remote, and push only after explicit approval.
5. Record the resulting GitHub coordinates:
   - repository: `<owner>/<repository>`
   - branch: `main` (or the reviewed release branch)
   - entrypoint: `app.py`

## 2. Deploy in Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with the GitHub account that can read the repository.
2. Select **Create app** and choose the option for an existing app/repository.
3. Enter the GitHub repository, reviewed branch, and file path `app.py`.
4. Open **Advanced settings** and select Python **3.12**.
5. Leave **Secrets** empty. The default app has no secret or API-key dependency.
6. Choose an app subdomain if desired, then select **Deploy**.
7. Watch the build logs. Confirm `requirements.txt` is detected and the app reaches the initial Decision Console without a missing-artifact exception.
8. Run the default K=4/p=1 scenario once, open Research Results, and check the desktop and narrow layouts.

Deployment URL placeholder: `https://<approved-subdomain>.streamlit.app`

## 3. Post-deployment checks

- Status reads **Validated Research MVP**.
- Default network preview renders without a network call.
- K=4/p=1 completes and a route decision appears.
- Research Results loads artifact-backed values rather than literals.
- Real topology cases load from committed `.graphml` files and remain labeled simulated traffic.
- Browser console has no errors.
- Pointer movement causes no WebSocket/network activity, Streamlit rerun, state mutation, or extra DOM node.
- Reduced-motion and coarse-pointer devices receive the static gradient treatment without pointer tracking.

## Troubleshooting

### Missing artifact

The app raises `FileNotFoundError`, `KeyError`, or a model-loading error when a required file was ignored or omitted.

1. Compare the deployed commit against the 64-file MVP set in `GITHUB_RELEASE_GUIDE.md`.
2. Check case-sensitive Linux filenames and forward-slash paths.
3. Confirm Git LFS objects, if used, were pushed and are available to the deployment.
4. Do not replace a missing validated artifact with a fabricated value. Restore the frozen file from the private archive.

### Dependency install failure

1. Confirm Community Cloud selected Python 3.12.
2. Confirm the root `requirements.txt` is the only detected app dependency file and contains the pinned runtime stack.
3. Inspect build logs for the first incompatible or unavailable package, not only the final resolver error.
4. Reproduce locally in a fresh Python 3.12 virtual environment. Do not loosen scientific package pins without validating the complete app and test suite.

### App memory pressure

1. Reboot the app to clear process memory and caches.
2. Use the K=4/p=1 default scenario; avoid K=8/p=3 and repeated dynamic quantum simulations during a public demo.
3. Confirm large checkpoint/archive files were not included in the deployment commit.
4. If pressure persists, inspect actual process usage before changing scientific behavior. Do not silently reduce or alter validated algorithms.

### Slow QAOA configuration

K=8/p=3 expands the analytic state space and circuit workload. Return to **LOAD DEMO SCENARIO** for deterministic K=4/p=1. Widget changes alone do not run QAOA; only form submission crosses the compute boundary. A completed scenario is cached by the full scenario configuration.

### Sleep and cold start

Community Cloud apps without traffic can hibernate. The next visitor may see a wake-up screen and must wake the app; model imports and first rendering can take longer than a warm rerun. This is hosting behavior, not QAOA execution. Keep the live demo open shortly before a presentation rather than adding artificial keep-alive traffic.

### Cache reset or stale state

- From the app menu, use **Clear cache** and rerun when a cached scenario or resource is stale.
- From the Streamlit workspace or app management logs, use **Reboot** for a clean process/build.
- Dependency changes trigger a full redeploy and may take longer than ordinary source updates.

## Local clean-start verification

From the repository root, use a fresh Python 3.12 environment:

```powershell
py -3.12 -m venv .venv-deploy-check
.\.venv-deploy-check\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.headless true
```

The directory is intentionally ignored by the `.venv/` pattern only when named `.venv`; remove or exclude `.venv-deploy-check` before staging. No public deployment is performed by this repository-preparation pass.
