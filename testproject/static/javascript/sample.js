window.onload = function() {
    var changeColor = function() {
        var e = document.getElementById('test');
        e.style.color = 'red';
        console.log("書き換えテスト")
    }
    setTimeout(changeColor, 5000);
}