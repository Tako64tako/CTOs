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

/*
const dress_back = document.getElementById("dress_back");
let dress_back_x = dress_back.clientWidth;
console.log(dress_back_x);*/
const sub_right = document.getElementById("sub_right");
const sub_left = document.getElementById("sub_left");
let sub_left_x = sub_left.clientWidth;


sub_right.scrollTop = sub_right.scrollHeight;

//左側背景のウィンドウサイズリサイズ時に動作する処理内容
//具体的にはウィンドウサイズが796を超えたとき、左側背景画像のwidthをウィンドウwidthによって決まる値にする
function resizeWindow(){
    sW = window.innerWidth;
    sub_left_x = sub_left.clientWidth;
    //console.log(sub_left_x)
    if(sub_left_x == 796){
        //console.log("calc"+(Number(sW) - Number(sub_left_x))+"px");
        sub_right.style.width= Number(sW) - Number(sub_left_x)-10+"px";
    }else{
        //console.log("kita")
        sub_right.style.width= "";
    }

}
//ウィンドウサイズリサイズ時に動作させる処理を登録
window.addEventListener('resize',resizeWindow);
//ウィンドウサイズリサイズ時に動作させる処理の実行
resizeWindow();//実行時にウィンドウサイズから要素のwidthを設定する


const icons = document.getElementsByClassName("icon");
let human_icon_flag = 0;
const asset_form = document.getElementById("asset_form");
const black_mask = document.getElementById("black_mask");
//暗くなった画面を明るくするための関数
function black_mask_hide(){
    if(black_mask.className != "hide_flag"){
        asset_form.className = "hide_flag";
        black_mask.className = "hide_flag";
        human_icon_flag = 0
    }
}

//右上の人物アイコンをクリックした時の関数
icons[0].addEventListener("click",function(){
    if(human_icon_flag==0){
        black_mask.addEventListener("mousedown",black_mask_hide);
        asset_form.className = "visible_flag";
        black_mask.className = "black_layer_4";
        human_icon_flag = 1
    }else{
        black_mask.removeEventListener("mousedown",black_mask_hide);
        asset_form.className = "hide_flag";
        black_mask.className = "hide_flag";
        human_icon_flag = 0
    }
});

const human_plus_icon = document.getElementById("human_plus_icon");
const file_form = document.getElementById("file_form");
//右上人物アイコンをクリックすると現れるフレーム内の左側のアイコンをクリックした時に動作させる関数
human_plus_icon.addEventListener("click",function(){
    black_mask.removeEventListener("mousedown",black_mask_hide);
    file_form.className = "visible_flag"
});

const human_already_icon = document.getElementById("human_already_icon");
const avater_select_form = document.getElementById("avater_select_form");
//右上人物アイコンをクリックすると現れるフレーム内の右側のアイコンをクリックした時に動作させる関数
human_already_icon.addEventListener("click",function(){
    black_mask.removeEventListener("mousedown",black_mask_hide);
    avater_select_form.className = "visible_flag"
    asset_form.className = "hide_flag"
});

const kyanseru = document.getElementsByClassName("kyanseru");
//左側アイコンをクリックすると現れるフレームのキャンセルボタンをクリックした時に動作させる関数
kyanseru[0].addEventListener("click",function(){
    file_form.className = "none_flag"
    avater_select_form.className = "none_flag"
    asset_form.className = "hide_flag";
    black_mask.className = "hide_flag";
    human_icon_flag = 0
});
//右側アイコンをクリックすると現れるフレームのキャンセルボタンをクリックした時に動作させる関数
kyanseru[1].addEventListener("click",function(){
    file_form.className = "none_flag"
    avater_select_form.className = "none_flag"
    asset_form.className = "hide_flag";
    black_mask.className = "hide_flag";
    human_icon_flag = 0
});

use_id = null//現在user_imgとなっている画像のid
function click_cut_img(e){
    console.log(e.id);
    user_img = document.getElementById("user_img");
    user_img.src = e.src;
    use_id = Number(e.id)
    user_img.className = "visible_flag";
    $.ajax({
        "url":click_cut_url,
        "type":"post",
        "data":{
            "use_id":use_id,
        },
        "dataType":"json"
    })
    .done(function(response){
        console.log("connect ok")
    });
}

const file_road_form = document.getElementById("id_picture");
//ファイルを選択したときにフォームに画像を表示させる処理(アップロードではない)
function fileChange(ev) {
    var target = ev.target;
    var file = target.files[0];
    var type = file.type; // MIMEタイプ
    var size = file.size; // ファイル容量（byte）
    var limit = 10000; // byte, 10KB
    console.log("type="+type)
    console.log("size="+size)
  
    // MIMEタイプの判定
    if ( !(type == 'image/jpeg'||type == 'image/png' )) {
        alert('選択できるファイルはJPEG画像とPNG画像だけです。');
        file_road_form.value = '';
        return;
    }
    /*
    // サイズの判定
    if ( limit < size ) {
      alert('10KBを超えています。10KB以下のファイルを選択してください。');
      inputFile.value = '';
    }*/

    const img = document.createElement("img");
    img.className = "road_img";
    let file_reader = new FileReader();
    file_reader.tmpImg = img;
    file_reader.onload = function () {
        this.tmpImg.src = this.result;
        this.tmpImg.onload = function () {
            document.getElementById('file_look').innerHTML = "";
            document.getElementById('file_look').appendChild(this);
        }
    }
    file_reader.readAsDataURL(file);
}
file_road_form.addEventListener("change", fileChange, false);

const load_forms = document.getElementsByClassName("load_form");
const hourglass = load_forms[0]
const counter = load_forms[1]
const cambus_form = document.getElementById("cambus_form")
const cambus = document.getElementById("cambus")
//画像ファイルを選択し、次へボタンを押したときajaxで画像ファイルを格納したDataFormを送信する関数を登録
$('#ajax-file-send').on('submit', function(e) {
    const start = Date.now();
    e.preventDefault();
    var fd = new FormData($("#ajax-file-send").get(0));
    for (let value of fd.entries()) {
        console.log(value);
    }
    black_mask.removeEventListener("mousedown",black_mask_hide);
    black_mask.className = "black_layer_9"//maskをつける
    hourglass.classList.remove("none_flag");//砂時計ロード動画を見せる

    $.ajax({
        'url': ajax_file_url,
        'type': 'post',
        'data': fd,
        'processData': false,
        'contentType': false
        /*'dataType': 'json'*/
    })
    .done(function(response){
        //Ajax通信が成功した場合に実行する処理
        const millis = Date.now() - start;
        console.log("millis:"+ millis + "[sec]")
        console.log("成功！");
        console.log(response);
        const select_img_contents = document.getElementById("select_img_contents");
        select_img_contents.innerHTML = ""
        human_img_array = response.queryset
        console.log(human_img_array)
        human_img_array.forEach( function( human_img_dict ) {
            console.log( human_img_dict );
            select_img_contents.innerHTML+='<div class="cut_img_contant"><img class="cut_img" src="'+human_img_dict.cut_image_path+'" id="'+human_img_dict.id+'" onclick="click_cut_img(this)"><h2 class="cut_img_name">'+human_img_dict.human_name+'</h2></div>';
        });
        /*
        for(i=0;i<human_img_array.length;i++){
            console.log(human_img_array[i])
            select_img_contents.innerHTML+='<div class="cut_img_contant" value="'+human_img_array[i].id+'"><img class="cut_img" src="'+human_img_array[i].cut_image_path+'"><h2 class="cut_img_name">'+human_img_array[i].human_name+'</h2></div>';
        }*/
        file_form.className = "none_flag"

        black_mask.className = "black_layer_4";
        black_mask.addEventListener("mousedown",black_mask_hide);
        human_icon_flag = 1

        avater_select_form.className = "visible_flag"
        asset_form.className = "hide_flag"

        hourglass.classList.add("none_flag");
    }).fail(function(data) {
        file_form.className = "none_flag"

        black_mask.className = "black_layer_4";
        black_mask.addEventListener("mousedown",black_mask_hide);
        human_icon_flag = 1

        avater_select_form.className = "visible_flag"
        asset_form.className = "hide_flag"

        hourglass.classList.add("none_flag");
    });
});

let co = 0;//画像の再読み込みに必要になる変数
//右の服画像を選択したときのイベントを登録する　着せ替え
const model_have = document.getElementsByClassName("model_have")
Array.prototype.forEach.call(model_have, function(element) {
    element.addEventListener("click",function(e){
        const start = Date.now();

        black_mask.removeEventListener("mousedown",black_mask_hide);
        black_mask.className = "black_layer_9"//maskをつける
        counter.classList.remove("none_flag");//蓄積ロード動画を見せる

        model_ele = e.children
        //console.dir(e.target)
        //console.dir(this)
        model_ele = this.children[0]

        model_data = model_ele.id//model_eleのidは"部位値.服名"のような形式になっている
        part_clothes = Number(model_data.split('.')[0])//どの部位の着せ替えをか
        clothes_name = model_data.split('.')[1]//服の名前
        console.log(part_clothes)
        console.log(clothes_name)
        console.log(use_id)
        if(use_id != null){
            $.ajax({
                "url":select_clothes_url,
                "type":"post",
                "data":{
                    "part_clothes":part_clothes,//どの部位の着せ替えをか
                    "clothes_name":clothes_name,//服の名前
                    "use_id":Number(use_id),//人物の番号
                },
                "dataType":"json"
            })
            .done(function(response){
                //Ajax通信が成功した場合に実行する処理
                const millis = Date.now() - start;
                console.log("millis:"+ millis + "[sec]")
                console.log("go back")
                console.log(response.chat)
                user_img = document.getElementById("user_img");

                //画像の再読み込みにおけるダミー引数案 この方法でないとブラウザ内のキャッシュにある画像を参照してしまい画像の再読み込みをしない
                co++;
                user_img.src = response.result_img_path+"?"+co;

                black_mask.addEventListener("mousedown",black_mask_hide);
                black_mask.className = "hide_flag"//maskを消す
                counter.classList.add("none_flag");//蓄積ロード動画を消す
            }).fail(function(data) {
                console.log("error!")
                black_mask.addEventListener("mousedown",black_mask_hide);
                black_mask.className = "hide_flag"//maskを消す
                counter.classList.add("none_flag");//蓄積ロード動画を消す
            });
        }else{
            console.log("use_id is null")
            black_mask.addEventListener("mousedown",black_mask_hide);
            black_mask.className = "hide_flag"//maskを消す
            counter.classList.add("none_flag");//蓄積ロード動画を消す
        }
    });
});
