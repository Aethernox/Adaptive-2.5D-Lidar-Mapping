# Contributing to Adaptive 2.5D LiDAR Mapping

Thank you for your interest in contributing! We welcome contributions to improve perception models, mapping performance, tracking algorithms, and visualization tools.

---

## Development Workflow

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/lidar_prototype_2.0.git
   cd lidar_prototype_2.0
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .\.venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes and Run Tests**:
   Ensure all tests pass before submitting changes:
   ```bash
   pytest tests/ -v
   ```

5. **Commit and Open a Pull Request**:
   - Write clear, concise commit messages.
   - Open a pull request against the `main` branch.

---

## Coding Standards

- Follow PEP 8 guidelines for Python.
- Maintain typing and explicit docstrings for all core contracts and algorithms.
- Add unit tests under `tests/` for newly introduced algorithms, transforms, or components.
