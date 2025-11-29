# Chatbot Documentation

Welcome to the Chatbot documentation! This documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## About MkDocs

MkDocs is a fast, simple, and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown and configured with a single YAML configuration file.

## Material Theme

This documentation uses the **Material for MkDocs** theme, which provides:

- Beautiful, responsive design with light/dark mode
- Advanced navigation features (tabs, sections, expandable menus)
- Powerful search functionality
- Code syntax highlighting and annotations
- Support for diagrams (Mermaid)
- **Admonitions** for callouts and notes

## Admonitions

Admonitions are special callout boxes that can be used to highlight important information, warnings, tips, and more. The Material theme supports various admonition types.

### Syntax

Admonitions use the following syntax:

```markdown
!!! type "Optional Title"
    Content of the admonition goes here.
    It can span multiple lines.
```

Or without a title:

```markdown
!!! type
    Content of the admonition goes here.
```

### Available Admonition Types

The following admonition types are configured in this documentation:

- `note` - General notes and information
- `abstract` - Summaries and abstracts
- `info` - Informational callouts
- `tip` - Helpful tips and tricks
- `success` - Success messages
- `question` - Questions or prompts
- `warning` - Warnings and cautions
- `failure` - Failure or error messages
- `danger` - Critical warnings
- `bug` - Bug reports or known issues
- `example` - Examples and demonstrations
- `quote` - Quotations and citations

### Examples

!!! note "Example Note"
    This is a note admonition with a title.

!!! tip
    This is a tip without a title. Tips are great for highlighting helpful information.

!!! warning "Important Warning"
    This is a warning admonition. Use it to draw attention to potential issues.

!!! success
    This indicates a successful operation or positive outcome.

!!! danger "Critical"
    Use danger admonitions for critical warnings that require immediate attention.

## Building Documentation

To build this project's documentation:

```bash
mkdocs build
```

This generates a static site in the `site/` directory.

## Serving Documentation Locally

To serve the documentation locally with live reload:

```bash
mkdocs serve
```

This starts a development server (typically at `http://127.0.0.1:8000`) that automatically reloads when you make changes to the documentation files.

## Configuration

Documentation configuration is managed in `mkdocs.yml` at the project root. This file controls:

- Site name, description, and author
- Theme settings and color palette
- Navigation structure
- Markdown extensions (including admonitions)
- Plugins (search, Mermaid diagrams, etc.)

## Learn More

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Admonitions Guide](https://squidfunk.github.io/mkdocs-material/reference/admonitions/)
