# Jiahui Chen Website

Personal academic website for Jiahui Chen, built with Jekyll and the
[`al-folio`](https://github.com/alshedivat/al-folio) theme.

Live site: [https://jiahuic.github.io](https://jiahuic.github.io)

## Local Development

```bash
bundle check || bundle install
bundle exec jekyll build
bundle exec jekyll serve
```

If port `4000` is already in use, run `bundle exec jekyll serve --port 4001`.

## Main Content Locations

- About page: `_pages/about.md`
- Publications list: `_bibliography/papers.bib`
- News items: `_news/`
- Projects: `_projects/`
- Teaching pages and course material: `_pages/teaching.md` and `_teaching/`
- CV data: `_data/cv.yml`
- Repository cards: `_data/repositories.yml`
- Site-wide settings: `_config.yml`

## Repository Layout

- `_pages/`: top-level pages in the navigation
- `_posts/`: blog posts
- `_projects/`: project cards and project detail pages
- `_teaching/`: course pages, lecture notes, and teaching resources
- `_news/`: short news and announcement entries shown on the homepage
- `assets/`: images, PDFs, JavaScript, and styles
- `.github/workflows/`: GitHub Actions deployment workflows

## Deployment

Pushes to `master` or `main` trigger the workflow in
[`deploy.yml`](.github/workflows/deploy.yml), which builds the site and deploys
the generated output to `gh-pages`.

## Theme Upgrades

This repository keeps an `upstream` remote pointing to
`alshedivat/al-folio`. Before pulling in upstream theme changes, review the
diff carefully because this site includes local content, navigation, and layout
customizations.
