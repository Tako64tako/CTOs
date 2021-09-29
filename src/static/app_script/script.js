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
    if(chat_form.text.value!=""){
        let add_text = '<div class="my_mes"><div class="cover"><a class="message">'+chat_form.text.value+'</a></div></div>';
        post_ele.innerHTML+=add_text;
        chat_data = chat_form.text.value;
        chat_form.text.value = ""
        //sectionの値ごとで質問し、sectionが3になったらサーバへデータを転送する
        switch(section){
            case 0:
                gender = chat_data;
                post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">身長？</a></div></div>';
                section++;
                break;
            case 1:
                height = chat_data;
                post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">種類？</a></div></div>';
                section++;
                break;
            case 2:
                type = chat_data;
                post_ele.innerHTML+='<div class="your_mes"><div class="cover"><a class="message">色？</a></div></div>';
                section++;
                break;
            case 3:
                color = chat_data;
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
        post_ele.scrollTop = post_ele.scrollHeight;
    }
}