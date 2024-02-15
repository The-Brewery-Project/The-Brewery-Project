# The Brewery Project

[The Brewery Project - Official Website](https://the-brewery-project.github.io/The-Brewery-Project/)

## General Procedures for Working with the Website

1. Install Quarto and `install.packages("rmarkdown")` if you haven't already (**VSCode** is an additional option to RStudio/Quarto).
2. Whenever you do git pull, always rebase: `git pull --rebase`.
3. If attempting something *major*, create a branch for any work that could have a major effect (aka the `.qmd` files), and start your branch name with initials, e.g. `CK_Branch_1`.
4. **ALWAYS** keep your git updated, do `git pull` always before any committing, pushing, branches, etc.
5. Recommended to perform `quarto render` on the CLI/bash before pushing.
6. When pushing, a force may be necessary. Make sure to pay attention to which branch is being pushed.
7. Github will automatically turn the `docs/` folder file changes into the website.

## Example Command Line and Work Procedure

1. cd into repository
2. `git pull --rebase`
3. pay attention to branch; will most likely be working on branch `main`
4. make contributions
5. it's good practice to make `git status` checks
6. optional, but recommended, apply `quarto render` 
7. `git add [file(s)]` (can use `git add .` to add all available files)
8. commit contribution: if text editor set to appear to receive and edit message, use `git commit`
9. commit contribution: (or alternatively), use `git commit -m "message"`
10. finally, `git push origin [branch]`; will most likely want to use `git push origin main`

## Data Sources

### [Open Brewery DB](https://www.openbrewerydb.org/)

- [GitHub Directory](https://github.com/openbrewerydb/openbrewerydb/)
  - [csv files](https://github.com/openbrewerydb/openbrewerydb/tree/master/data)
- [API Documentation](https://www.openbrewerydb.org/documentation)
- [openbrewerydb-python (*Python API Wrapper Module*)](https://jrbourbeau.github.io/openbrewerydb-python/)
  - [Example Usage](https://jrbourbeau.github.io/openbrewerydb-python/examples.html)
