# Precomputed Demonstration Data

The files in this directory (`demo_results.csv` and `demo_live_status.json`) are intentionally frozen, precomputed datasets generated from the local SUMO microscopic traffic simulation.

**Why are these here?**
They are specifically used by the **Public Demo Mode** in the Streamlit dashboard to visually demonstrate the capabilities of the Traffic Detection & Optimization System without requiring a live instance of the heavy SUMO C++ binaries on the hosting cloud server.

**Note on Live Simulation:**
These files are **not** live. The live SUMO simulation writes its actual runtime state into `01_Pro_Version/data/`. The local research environment dynamically polls the `data` directory, while the public cloud demo relies on this static `demo_data` directory to guarantee a lightweight, dependency-free experience for web visitors.
