# 🚀 Streamlit Cloud Deployment Guide: BioRetrosynthesis

## Quick Deployment (2 minutes)

### Step 1: Repository Details
- Repository: **https://github.com/mayurOG/BIORETROSYNTHESIS-.git**  
- Branch: `main`

### Step 2: Deploy on Streamlit Community Cloud

1. **Open Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account (`mayurOG`)

2. **Click "Create app" / "New app"**

3. **Configure App Settings:**
   - **Repository:** `mayurOG/BIORETROSYNTHESIS-`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py` *(or `net_app_advanced.py`)*
   - **App URL:** (Optional) customize your subdomain, e.g. `bioretrosynthesis-app`

4. **Click "Deploy!"**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt` and system graphics libraries from `packages.txt`.
   - Within 1–2 minutes, your live web application will be active and publicly accessible!

---

## 🛠 Deployment Configuration

- **`streamlit_app.py`**: Streamlit Community Cloud default entrypoint.
- **`net_app_advanced.py`**: Main application code with full interactive visualization.
- **`requirements.txt`**: Optimized production dependencies for Streamlit Cloud (`streamlit`, `rdkit`, `plotly`, `pandas`, `networkx`, `pubchempy`).
- **`packages.txt`**: Debian graphics libraries (`libxrender1`, `libglib2.0-0`, `libsm6`, `libxext6`) required by RDKit in headless Linux environments.
- **`.streamlit/config.toml`**: Streamlit theme and server configuration.

---

## 💻 Running Locally

```bash
streamlit run streamlit_app.py
```
Open **http://localhost:8501** in your browser.
