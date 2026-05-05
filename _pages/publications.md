---
layout: page
permalink: /publications/
title: Publications
description: Journal articles, conference papers, and recent preprints.
years: [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2015]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
See the [Projects]({{ '/projects/' | relative_url }}) page for related software and application details.

<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
