---
layout: post
title:  Springboot JPA
summary: "Springboot/JPA"
author: jiyun
date: '2021-01-28 15:35:23 +0530'
category: springboot
thumbnail: /assets/img/posts/springboot.png
keywords: springboot, jpa  
permalink: /blog/springboot-JPA/
usemathjax: true
--- 


>## JPA   
>#### 서로 지향하는 바가 다른 (객체지향, 관계형 데이터베이스) 2개 영역을 중간에서 일치 시켜주기 위한 기술   
>#### 개발자가 sql에 종속적인 개발을 하지 않아도 된다.   

<br>

>### Spring Data JPA   
>#### 인터페이스인 JPA를 사용하기 위한 구현체가 필요하다. (Hibernate)   
>#### Spring에서는 구현체들을 더 쉽게 사용하기 위해 Spring Data JPA 모듈을 이용한다.   

<br>
    
- Spring Data JPA   
  
        <관계>
  
        JPA  <-- Hibernate  <-- Spring Data JPA   

    - 구현체 교체가 용이하다.   
        `Hibernate말고 다른 구현체로의 교체가 자유롭다.`
      
    - 저장소 교체가 용이하다.    
        `관계형 데이터베이스 외에 다른 저장소로 쉽게 교체가 가능하다.`   

<br>
<br>


>#### JPA 사용 예시   

- @Entity   
`테이블과 링크될 클래스임을 나타낸다.`   
`기본값으로 클래스의 카멜케이스 이름을 언더스코어 네이밍으로 매칭한다.`   


- @Id   
`해당 테이블의 PK필드를 나타낸다.`   
  

- @GeneratedValue   
`PK의 생성규칙을 나타낸다.`   
  `스프링부트 2.0 부터 GenerationType.IDENTITY옵션을 추가해야 auto_increment를 설정할 수 있다.`   
  

- @Column   
`테이블의 칼럼을 나타낸다. (기본값 외에 추가로 변경이 필요한 옵션이 있으면 사용.)`   


- @NoArgsConstructor   
`기본 생성자 자동 추가 (public Posts(){} 와 같음)`   
  
  
- @Getter   
`클래스 내 모든 필드의 Getter 메소드를 자동생성`   
  

- @Builder   
`해당 클래스의 빌더 패턴 클래스를 생성`   
  


```java
@Getter //
@NoArgsConstructor //
@Entity //
public class Posts {
    @Id //
    @GeneratedValue(strategy = GenerationType.IDENTITY) //
    private Long id;

    @Column(length = 500, nullable = false) //
    private String title;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String content;

    private String author;

    @Builder //
    public Posts(String title, String content, String author){
        this.title = title;
        this.content = content;
        this.author = author;
    }

}
```

<br>

*Entity의 PK는 long타입의 auto_increment가 PK를 단순하게 나타낼 수 있어서, 복잡한 상황을 피할 수 있다.*   

*또, Entity 클래스와 Entity Repository는 함께 위치해야 한다.*   

<br>


**테스트 코드**   

- @After   
`JUnit에서 단위 테스트가 끝날 때마다 수행되는 메소드를 지정`   
  `배포 전 전체 테스트를 수행할 때 테스트간 데이터 침범을 막기 위해 사용.`   
  

- postsRepository.save   
`테이블 posts에 insert/update 쿼리 실행 (id 값이 있으면 update, 없으면 insert)`   
  

- postsRepository.finaAll   
`테이블 posts에 있는 모든 데이터를 조회해오는 메소드.`   
  

```java
@RunWith(SpringRunner.class) 
@SpringBootTest
public class PostsRepositoryTest {

    @Autowired
    PostsRepository postsRepository;

    @After //
    public void cleanup() {
        postsRepository.deleteAll();
    }

    @Test
    public void 게시글저장_불러오기() {
        String title = "테스트 게시글";
        String content = "테스트 본문";

        postsRepository.save(Posts.builder() //
        .title(title)
        .content(content)
        .author("jiyun@naver.com")
        .build());

        List<Posts> postsList = postsRepository.findAll(); //

        Posts posts = postsList.get(0);
        assertThat(posts.getTitle()).isEqualTo(title);
        assertThat(posts.getContent()).isEqualTo(content);
    }
}
```   
<br>
<br>


>### Spring 웹 계층   

<br>

<img src="https://user-images.githubusercontent.com/52434993/107610831-e4cca880-6c85-11eb-85c4-40e1c3d0599d.png" width = "60%">    

- Web Layer   
`흔히 사용하는 컨트롤러, JSP 등의 뷰 템플릿 영역`   
  `외부 요청과 응답에 대한 전반적인 영역`   
  

- Service Layer   
`Controller와 DAO의 중간 영역.`   
  `@Transactional이 사용되어야 하는 영역`   
  

- Repository Layer   
`Database와 같이 데이터 저장소에 접근하는 영역 (DAO 영역)`   
  

- DTOs   
`계층 간에 데이터 교환을 위한 객체.`   
  

- Domain Model   
`도메인이라 불리는 개발 대상을 모든 사람이 동일한 관점에서 이해할 수 있고 공유할 수 있도록 단순화 시킨 것.`   
`@Entity가 사용된 영역`   
  

<br><br>

>### 클라이언트에서 데이터 저장시 발생하는 일의 순서   
>#### Entity에 데이터를 저장하기 위해서, Entity 클래스의 정보를 서버에 request 요청을 보낸다.
>#### 이후 서버에서는 DTO 클래스를 이용해서 Entity 필드의 값을 비즈니스 로직을 처리하는 곳으로 정보를 넘긴다.   
>#### 여기서 비즈니스 로직을 처리하는 곳은 도메인 모델을 뜻한다.   


#### *이후에 JPA에 대한 자세한 공부를 할 것.*   

