//CSRFトークンの処理
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

$.ajaxSetup({
    beforeSend: function (xhr, settings) {
        if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});

//main
const chat_form = document.getElementById("chat_form");
const post_ele = document.getElementById("post_ele");
let chat_data = "";
let gender,height,type,color;
section = 0;
chat_form.onsubmit = function(event){//送信ボタンを押したときの処理
    event.preventDefault();
    console.log("a");
    // if(chat_form.text.value!=""){
    //     let add_text = '<div class="my_mes"><div class="cover"><a class="message">'+chat_form.text.value+'</a></div></div>';
    //     post_ele.innerHTML+=add_text;
    //     chat_data = chat_form.text.value;
    //     chat_form.text.value = ""

    switch(section){//sectionの値ごとで質問し、sectionが4になったらサーバへデータを転送する
      case 0:
      console.log("b");
          let str = "";
          const color1 = document.form1.color1;
          console.log(color1)

          // for (let i = 0; i < color1.length; i++) {
          if (color1[0].checked == true) {//(color1[i].checked === true)と同じ
            str = color1[0].value;
            let add_text = '<div class="my_mes"><div class="cover"><a class="message">'+str+'</a></div></div>';
            post_ele.innerHTML+=add_text;
            gender = str;
            // break;
          }else if(color1[1].checked == true){
            str = color1[1].value;
            add_text = '<div class="my_mes"><div class="cover"><a class="message">'+str+'</a></div></div>';
            post_ele.innerHTML+=add_text;
            gender = str;
          }
          console.log(gender)
          post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">身長？</a></div></div>';
          chat_form.innerHTML='<input type="number" id="number" value="100" min="100" max="200" step="1"><input type="submit" value="送信！" id="send_butt">'
          section++;
          break;
      case 1:
          console.log("c");
          //post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">身長？</a></div></div>';
          //chat_form.innerHTML='<input type="number" id="number" value="100" min="100" max="200" step="1"><input type="submit" value="送信！" id="send_butt">'
          const number = document.getElementById("number");
          add_text = '<div class="my_mes"><div class="cover"><a class="message">'+number.value+'</a></div></div>';
          post_ele.innerHTML+=add_text;
          height = number.value;
          console.log(height)
          post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">種類？</a></div></div>';
          chat_form.innerHTML='<input type="text" name="text" id="chat_box"><input type="submit" value="送信！" id="send_butt">'
          section++;
          break;
      case 2:
          //height = chat_data;
          //post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">種類？</a></div></div>';
          add_text = '<div class="my_mes"><div class="cover"><a class="message">'+chat_box.value+'</a></div></div>';
          post_ele.innerHTML+=add_text;
          type = chat_box.value;
          console.log(type);
          chat_form.text.value = ""

          post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">色？</a></div></div>';
          section++;
          break;
      case 3:
          //type = chat_data;
          //post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">色？</a></div></div>';
          add_text = '<div class="my_mes"><div class="cover"><a class="message">'+chat_box.value+'</a></div></div>';
          post_ele.innerHTML+=add_text;
          color = chat_box.value;
          console.log(color);

          //color = chat_data;
          $.ajax({
              "url":send_customer_url,
              "type":"post",
              "data":{
                  "gender":gender,
                  "height":height,
                  "type":type,
                  "color":color,
              },
              "dataType":"json"
          })
          .done(function(response){
              post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">'+response.chat+'</a></div></div>';
              post_ele.scrollTop = post_ele.scrollHeight;
          });
          section++;
          break;
    }

        //sectionの値ごとで質問し、sectionが3になったらサーバへデータを転送する
        // switch(section){
        //     case 1:
        //         post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">身長？</a></div></div>';
        //         chat_form.innerHTML='<input type="number" id="number" value="100" min="100" max="200" step="1"><input type="submit" value="送信！" id="send_butt">'
        //         const number = document.getElementById("number");
        //         height = number;
        //         section++;
        //         break;
        //     case 1:
        //         height = chat_data;
        //         post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">種類？</a></div></div>';
        //
        //         section++;
        //         break;
        //     case 2:
        //         type = chat_data;
        //         post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">色？</a></div></div>';
        //         section++;
        //         break;
        //     case 3:
        //         color = chat_data;
        //         $.ajax({
        //             "url":send_customer_url,
        //             "type":"post",
        //             "data":{
        //                 "gender":gender,
        //                 "height":height,
        //                 "type":type,
        //                 "color":color,
        //             },
        //             "dataType":"json"
        //         })
        //         .done(function(response){
        //             post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">'+response.chat+'</a></div></div>';
        //             post_ele.scrollTop = post_ele.scrollHeight;
        //         });
        //         section++;
        //         break;
        // }
        post_ele.scrollTop = post_ele.scrollHeight;
    }
