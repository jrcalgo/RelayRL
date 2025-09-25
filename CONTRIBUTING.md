# Contributing to RelayRL

Thank you for your interest in contributing to RelayRL! This guide will help you make high-quality contributions and understand our development workflow.


## Code of Conduct

By participating, you are expected to uphold the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).


## How to Contribute

### 1. Fork & Clone
- Fork the repository to your own GitHub account.
- Clone your fork locally:
  ```sh
  git clone https://github.com/your-username/RelayRL-prototype.git
  cd RelayRL-prototype
  ```

### 2. Create a Branch
- Always create a new branch for your work:
  ```sh
  git checkout -b feature/your-feature-name
  ```

### 3. Code Style & Best Practices
- **Rust:**
  - Use `rustfmt` for formatting (`cargo fmt`).
  - Run `cargo clippy` for linting and fix warnings where possible.
- **Python:**
  - Use [black](https://black.readthedocs.io/en/stable/) for formatting.
  - Run [flake8](https://flake8.pycqa.org/en/latest/) for linting.
- **General:**
  - Write clear, concise, and well-documented code.
  - Add or update docstrings and comments as needed.
  - Include or update tests if applicable.

### 4. Commit Messages
- Use clear, descriptive commit messages.
- Follow the format:
  ```
  <type>: <short summary>

  [optional body]
  [optional footer]
  ```
  - **type**: feat, fix, docs, style, refactor, test, chore, etc.
  - Example:
    ```
    feat: add support for custom reward functions
    ```

### 5. Pull Requests (PRs)
- Push your branch to your fork:
  ```sh
  git push origin feature/your-feature-name
  ```
- Open a PR against the `main` branch of the upstream repository.
- Fill out the PR template (if available) and describe your changes clearly.
- Link related issues or discussions if applicable.
- Mark the PR as a draft if it is not ready for review.

### 6. Review Process
- PRs will be reviewed by maintainers and/or other contributors.
- Address feedback and make changes as requested.
- All checks (CI, tests, formatting, linting) must pass before merging.
- Squash or rebase commits if requested.

### 7. Merging
- Only maintainers can merge PRs.
- PRs should be up-to-date with the `main` branch before merging.


## Communication
- Use [GitHub Issues](https://github.com/your-org/RelayRL-prototype/issues) for bug reports, feature requests, and questions.
- For larger changes, open a discussion or issue before starting work.


## Additional Notes
- This project is a **prototype** and is **unstable during training**. Expect rapid changes and breaking updates.
- Please be respectful and constructive in all communications.


Thank you for helping make RelayRL better! 
