Dev container for rkllama

How to open:

- In VS Code run: "Remote-Containers: Reopen in Container" (or "Dev Containers: Reopen in Container").

What runs after the container builds:

- The `postCreateCommand` will try to install Python dependencies from `converter/requirements.txt` and install the project in editable mode (`pip install -e .`).

Notes:

- The devcontainer uses the repository Dockerfile at the workspace root. If that Dockerfile requires additional local artifacts, build may need extra privileges or network access.
- If dependency installation fails (for platform-specific wheels), install them manually inside the container.
