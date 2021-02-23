---
layout: post
title:  Springboot setting
summary: "Springboot/gradle"
author: jiyun
date: '2021-01-15 15:35:23 +0530'
category: springboot
thumbnail: /assets/img/posts/springboot.png
keywords: springboot, jpa  
permalink: /blog/springboot-gradle/
usemathjax: true
--- 



>## 그레이들 프로젝트 스프링부트 프로젝트로 변경   
>#### build.gradle파일을 수정.

<br>

*스프링 이니셜라이저 사용대신 하나하나 설정*   

*build.gradle코드 일부*   

```
buildscript{

    // build.gradle에서 사용하는 전역변수를 설정하는 부분.
    ext {
       
        // springBootVersion 전역변수 생성.
        // 스프링 부트 그레이들 플러그인의 2.1.7.RELEASE를 의존성으로 받겠다.
        springBootVersion = '2.1.7.RELEASE'
    }
    
    // 각종 의존성들을 어떤 원격 저장소에서 받을지,
    // 최근에는 jcenter가 대
    repositories{
        mavenCentral()
        jcenter()
    }
    dependencies{
        classpath("org.springframework.boot:spring-boot-gradle-plugin:${springBootVersion}")
    }
}
```
