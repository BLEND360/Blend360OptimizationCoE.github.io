# Optimization CoE website v1.0

The optimization CoE website is built using the [**Minimal Mistake**](https://mmistakes.github.io/minimal-mistakes/) Jekyll theme which is built in *Ruby* and is supported by [Michael Rose](https://github.com/mmistakes). The theme is free and will be updated automatically (caveat: theme overrides will not be updated and we should update them manually). The CoE website theme is mostly based on the original theme with modifications to layout structure and pagination:
  - author: added support for multiple authors per post (see _includes/author-profile.html)
  - head: added support for compiling LateX markdown using mathjax (see _includes/head.html)
  - single: added support for longer excerpts (see _includes/archive-single.html)
  - team page: added a page introducing the team (see _team/ and _data/navigation.yml)

## Post new content to website

Jekyll is able to digest a variety of markdown notebooks as well as html pages. To use the themes capabilities such as table of content, multiple authors, math formula, each markdown notebook should follow a certain structure.

- **post document name**: Each post's name should start with the date in the following format `YYYY-mm-dd` plus the name of the document. You can refer to the _post/ folder in this git repo and check out two sample posts; [2021-08-11-optimal_staff_level_job_shop_scheduling](https://github.com/BLEND360/Blend360OptimizationCoE.github.io/edit/master/_posts/2021-08-11-optimal_staff_level_job_shop_scheduling.md) and [2021-11-18-dynamic_moving_policy_with_error](https://github.com/BLEND360/Blend360OptimizationCoE.github.io/edit/master/_posts/2021-11-18-dynamic_moving_policy_with_error.md).

- **post YAML front matter**: Each post document should include a YAML front matter denoted by `--- ---` which can include the following
  - title
  - excerpt: summary introduction of the post limited to 1000 characters
  - toc: if <true>, includes a table of content based on markdown header structure
  - toc_label: table of content label
  - toc_sticky: if <true>, the table of content will stick to the page when scrolling
  - author (authors): Use <authors> when having multiple authors. Each author's name should be included in the _data/authors.yml file. See [2021-08-11-optimal_staff_level_job_shop_scheduling](https://github.com/BLEND360/Blend360OptimizationCoE.github.io/edit/master/_posts/2021-08-11-optimal_staff_level_job_shop_scheduling.md) for an example of a post with one author and [2021-11-18-dynamic_moving_policy_with_error](https://github.com/BLEND360/Blend360OptimizationCoE.github.io/edit/master/_posts/2021-11-18-dynamic_moving_policy_with_error.md) for an example with two authors. If assigning no authors, the site default authors, Blend360 will be used.
  - teaser (subitem of header): Add the absolute path of a teaser image which shows up in the `YOU MAY ALSO ENJOY` section at the bottom of each post

- **categories**: To categorize a post, use the YAML front matter according to one of the examples and categorize the post into one of the following categories
  - Solution notebook; these posts should contain model and code information
  - White paper; these posts may only include a model

- **tags**: Each post can also be tagged in different ways. It is best to keep the number of tags limited, however, generally, tags can classify from different aspects and are more granular than categories. Tags will also be assigned in the YAML front matter.

- **images**: To add an image to the body of a post, put the image in assets/images/Post-images/<post doc name>/<image file name with no spaces>. Including an image in a post can be done with both markdown and html syntax. However, it is recommended to use the following to handle absolute image paths `{{ site.url }}{{ site.baseurl }}/assets/images/Post-images/{{ page.path | replace: '.md', '' | replace: '_posts/', '' }}/<image file name with no spaces>.`

- **continuous headers for the toc**: To take advantage of the automatically generated table of content, markdown headers should be continuous and not skip a step. Markdown headers are usually identified by `#`, `##`, `###`, and so on based on their hierarchy. Skipping one of these levels will mess up the table of content.

- **alignment and notices**: These are some of the text modification utilities that the Minimal Mistake theme provides. To learn how to use them, refer to the [Utility section](https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/) of the theme documentation.

- **math formula**: To include math formula with LateX in your markdown, use `usemathjax: true` in the YAML front matter of the post. For inline math, use $$<math>$$. For equations, use a blank line before and after the $$<equation>$$.

- **converting .tex to .md**: Use cloud converter at [https://cloudconvert.com/](https://cloudconvert.com/) to convert LateX files to markdown. Depending on the level of detail, some converted markdown files will need proofing. Note that, the headers in the converted markdown would use the `==`, `--` syntax which is compatible with the theme. However, it only supports two levels. To add more levels to the hierarchy of headers, use the `#`, `##`, `###` syntax.

- **converting bibliography (.bib) to .md**: For references, you can copy the rendered result in the markdown. For extensive bibliographies, use [pandoc](https://pandoc.org/installing.html) to convert .bib files to markdown according to this [StackOverflow](https://tex.stackexchange.com/questions/610876/generating-markdown-from-bib-file?utm_source=pocket_mylist) post.

## Adding new author and team members

To add a team member to the site's [team page](https://blend360.github.io/Blend360OptimizationCoE.github.io/team/), create a markdown similar to the content of _team/ folder. To add a new author for the website, configure the _/data/authors.yml file.

## For any issues
Contact [Amir Nasr](amir.nasrollahzadeh@blend360.com).
