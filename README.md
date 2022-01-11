# Optimization CoE website v1.0

The optimization CoE website is built using the [**Minimal Mistake**](https://mmistakes.github.io/minimal-mistakes/) Jekyll theme which is built in *Ruby* and is supported by [Michael Rose](https://github.com/mmistakes). The theme is free and will be updated automatically (caveat: theme overrides will not be updated and we should update them manually). The CoE website theme is mostly based on the original theme with modifications to layout structure and pagination:
  - author: added support for multiple authors per post (see _includes/author-profile.html)
  - head: added support for compiling LateX markdown using mathjax (see _includes/head.html)
  - single: added support for longer excerpts (see _includes/archive-single.html)
  - team page: added a page introducing the team (see _team/ and _data/navigation.yml)

## Post new content to the website

Jekyll is able to digest a variety of markdown notebooks as well as html pages. To use the themes capabilities such as table of content, multiple authors, math formula, each markdown notebook should follow a certain structure.

- **post document name**: Each post's name should start with the date in the following format 'YYYY-mm-dd' plus the name of the document. You can refer to the _post/ folder in this git repo and check out two sample posts; '2021-08-11-optimal_staff_level_job_shop_scheduling' and '2021-11-18-dynamic_moving_policy_with_error'.

- **post YAML front matter**:

Click [**Use this template**](https://github.com/mmistakes/mm-github-pages-starter/generate) button above for the quickest method of getting started with the [Minimal Mistakes Jekyll theme](https://github.com/mmistakes/minimal-mistakes).

Contains basic configuration to get you a site with:

- Sample posts.
- Sample top navigation.
- Sample author sidebar with social links.
- Sample footer links.
- Paginated home page.
- Archive pages for posts grouped by year, category, and tag.
- Sample about page.
- Sample 404 page.
- Site wide search.

Replace sample content with your own and [configure as necessary](https://mmistakes.github.io/minimal-mistakes/docs/configuration/).

---

## Troubleshooting

If you have a question about using Jekyll, start a discussion on the [Jekyll Forum](https://talk.jekyllrb.com/) or [StackOverflow](https://stackoverflow.com/questions/tagged/jekyll). Other resources:

- [Ruby 101](https://jekyllrb.com/docs/ruby-101/)
- [Setting up a Jekyll site with GitHub Pages](https://jekyllrb.com/docs/github-pages/)
- [Configuring GitHub Metadata](https://github.com/jekyll/github-metadata/blob/master/docs/configuration.md#configuration) to work properly when developing locally and avoid `No GitHub API authentication could be found. Some fields may be missing or have incorrect data.` warnings.
