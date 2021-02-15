---
layout: page
title: ML_guide
permalink: /blog/categories/ML_guide/
---

<h5> Posts by Category : {{ page.title }} </h5>

<div class="card">
{% for post in site.categories.ML_guide %}
 <li class="category-posts"><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a>
 <img src="../assets/img/authors/jiyun.jpg" width="25px" height="25px"/>
 </li>
{% endfor %}
</div>
