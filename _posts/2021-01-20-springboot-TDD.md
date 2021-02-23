---
layout: post
title:  Springboot TDD(Test-driven development)
summary: "Springboot/TDD"
author: jiyun
date: '2021-01-20 15:35:23 +0530'
category: springboot
thumbnail: /assets/img/posts/springboot.png
keywords: springboot, jpa, TDD  
permalink: /blog/springboot-TDD/
usemathjax: true
--- 


>## TDD (Test-driven development) / 단위 테스트 (Unit Test) 
>#### 테스트가 주도하는 개발.   

<br>   

- TDD
    - 레드 그린 사이클   

        <img src ="https://user-images.githubusercontent.com/52434993/107529625-67118a00-6bfe-11eb-99d2-1b00acb3c368.gif"  width = 60%>
    
    - 항상 실패하고 테스트를 먼저 작성하고 (red)   
    
    - 테스트가 통과하는 프로덕션 코드를 작성하고 (green)   
    
    - 테스트가 통과하면 프로덕션 코드를 리팩토링합니다 (refactor)    

  
<br>

- 단위 테스트     

    - 개발단계 초기에 문제를 발견하게 한다.
    - 코드를 리팩토링하거나 라이브러리 업그레이드 등에서 기존 기능이 올바르게 작동하는지 확인할 수 있다.   
    - 시스템에 대한 실제 문서를 제공한다.  
    

<br>

*java의 경우에는 JUit을 활용한다.*   

<br><br>

>### 실제 단위 테스트 예시 1   
>#### Hello클래스를 만들고, 이를 실행하는 controller를 만든다.   
>#### 이후 제대로 실행되고 있는지에 대한 test코드를 만든다.

<br>

*controller 코드*   

- @RestController   
`컨트롤러를 JSON을 반환하는 컨트롤러로 만든다.`   
`@ResponseBody보다 더 편하다. (한번에 사용할 수 있게 한다.)`   
  
- @GetMapping   
`HTTP Method인 Get의 요청을 받을 수 있는 API를 만든다.`   

```java
@RestController
public class HelloController {

    @GetMapping("/hello")
    public String hello(){
        return "hello";
    }
}
```   

*Test코드*   

- @RunWith(SpringRunner.class)   
`테스트를 진행할 때 JUnit에 내장된 실행자 외에 다른 실행자를 실행`   
`(SpringRunner 실행자 사용.)`   
`스프링부트 테스트와 JUnit사이에 연결자 역할`
  
- @WebMvcTest   
`Web(Spring MVC)에 집중할 수 있는 어노테이션`   
`@Controller, @ControllerAdvice 등을 사용할 수 있다.`   

- @Autowired   
`스프링이 관리하는 Bean을 주입받는다.`
  
- private MockMvc mvc   
`웹 api를 테스트할 때 사용한다.`   
`스프링 MVC 테스트의 시작점`   
`HTTP GET, POST등에 대한 API테스트를 할 수 있다.`   
  
- mvc.perform(get("/hello"))   
`MockMvc를 통해 /hello 주소로 HTTP GET 요청을 한다.`   

- .andExpect(status().isOk())   
`mvc.perform의 결과를 검증한다.`
`HTTP Header의 Status를 검증한다.`   
`isOk이므로 200인지 아닌지를 검증한다.`   
  
- .andExpect(content().string(hello))   
`응답 본문의 내용을 검증한다.`
`Controller에서 "hello"를 리턴하기 때문에 이 값을 검증한다.`   
     
```java
@RunWith(SpringRunner.class) //
@WebMvcTest(controllers = HelloController.class) //
public class HelloControllerTest {
    @Autowired //
    private MockMvc mvc; //

    @Test
    public void hello가_리턴된다() throws Exception {
        String hello = "hello";

        mvc.perform(get("/hello")) //
                .andExpect(status().isOk()) //
                .andExpect(content().string(hello)); //
    }
}
```   

<br><br>


>## 실제 단위 테스트 예시 2(feat. lombok)   
>#### 테스트 코드를 lombok으로 변경하고, 실행.    

<br>

*lombok을 이용한 getter 클래스 파일*   

- @Getter    
`선언된 모든 필드의 get메소드를 생성해 준다.`     
  
  
- @RequiredArgsConstructor   
`선언된 모든 final 필드가 포함된 생성자를 생성해 준다.`   
`final이 없는 필드는 생성자에 포함되지 않는다.`     
  

```java
@Getter
@RequiredArgsConstructor
public class HelloResponseDto {
    private final String name;
    private final int amount;
}
```   

*lombok 테스트 파일*   

- assertThat   
`검증하고 싶은 대상을 메소드 인자로 받고, isEqualTo와 같이 메소드를 이어서 사용가능하다.`   
  


```java
public class HelloResponseDtoTest {

    @Test
    public void 롬복_기능_테스트(){
        String name = "test";
        int amount = 1000;

        HelloResponseDto dto = new HelloResponseDto(name, amount);

        assertThat(dto.getName()).isEqualTo(name);
        assertThat(dto.getAmount()).isEqualTo(amount);
    }
}
```   

<br>

*처음의 HelloController에 새로 만든 lombok class 추가.*      

- param   
`API테스트할 때 사용될 요청 파라미터를 설정한다.`   
`값은 String만 허용된다.`   
  

- jsonPath   
`JSON 응답값을 필드별로 검증할 수 있는 메소드이다.`   
`$ 를 기준으로 필드명을 명시한다.`    
  
  
```java

--- lombok 테스트 코드 추가 부분 ---

@Test
    public void helloDto가_리턴된다() throws Exception {
        String name = "hello";
        int amount = 1000;

        mvc.perform(
                get("/hello/dto")
                        .param("name", name)
                        .param("amount", String.valueOf(amount)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.name", is(name)))
                .andExpect(jsonPath("$.amount", is(amount)));

    }
```
