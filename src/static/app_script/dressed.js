const dress_back = document.getElementById("dres_back");
let dress_back_x = dress_back.clientWidth;
console.log(dress_back_x);
const sub_right = document.getElementById("sub_right");
const sub_left = document.getElementById("sub_left");
let sub_left_x = sub_left.clientWidth;

sub_right.scrollTop = sub_right.scrollHeight;

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
//ウィンドウサイズを変えると実行される
window.addEventListener('resize',resizeWindow);
resizeWindow();//実行時にウィンドウサイズから要素のwidthを設定する

const icons = document.getElementsByClassName("icon");
let human_icon_flag = 0;
const asset_form = document.getElementById("asset_form");
const black_mask = document.getElementById("black_mask");

function black_mask_hide(){
    if(black_mask.className != "hide_flag"){
        asset_form.className = "hide_flag";
        black_mask.className = "hide_flag";
        human_icon_flag = 0
    }
}
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
human_plus_icon.addEventListener("click",function(){
    black_mask.removeEventListener("mousedown",black_mask_hide);
    file_form.className = "visible_flag"
});

const kyanseru = document.getElementsByClassName("kyanseru");
kyanseru[0].addEventListener("click",function(){
    file_form.className = "hide_flag"
    asset_form.className = "hide_flag";
    black_mask.className = "hide_flag";
    human_icon_flag = 0
})

const file_road_form = document.getElementById("uploadfile");
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

const cambus_form = document.getElementById("cambus_form")
const cambus = document.getElementById("cambus")

  $('#ajax-file-send').on('submit', function(e) {
    e.preventDefault();
    var fd = new FormData($("#ajax-file-send").get(0));
    for (let value of fd.entries()) { 
        console.log(value); 
    }
    $.ajax({
        'url': ajax_file_url,
        'type': 'post',
        'data': fd,
        'processData': false,
        'contentType': false,
        'dataType': 'json'
    })
    .done(function(response){
        //Ajax通信が成功した場合に実行する処理
        console.log("ppap");
        /*
        file_form.className = 'none_flag';
        cambus_form.className = 'visible_flag';
        img_html_txt = "<img src='"+response.img_url+"' alt='no_img' id='cut_out_img'>";
        cambus.innerHTML = img_html_txt;*/
    });
});