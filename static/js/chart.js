function plot_faces(faces_data){
    for (let x in faces_data)
    {
        // let face_url = "http://127.0.0.1:8000/static/result/person/p_" + String(x) + ".jpg";
        let face_url = "http://localhost:8888/static/result/person/p_" + String(x) + ".jpg";
        // let div = document.createElement("div");
        let img = document.createElement("img");
        img.className = 'face_pic';
        img.id = 'face_' + String(x);
        img.src = face_url;
        $(".faces").append(img);
        face_id = '#face_' + String(x);
        $(face_id).on('click',function(){
            // window.location.href="http://127.0.0.1:8000/face_stat/?face_id=" + String(x);
            window.location.href="http://localhost:8888/face_stat/?face_id=" + String(x);
        })

    }
}