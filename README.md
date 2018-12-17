# Facial Expression Recognition

#### 1. Download model

https://drive.google.com/open?id=1BhKRwCX2g96LTiGAm3VtsnnFAYJnUF9J

Please down this file at `./src/model/face_recognition/model_data/`

#### 2. Install python package

At first, you need install these package, as bellow:

 `pip install -r requirements.txt `

Requirements includes:

django
numpy
pandas
opencv-python
Pillow
tensorflow
keras
scipy

#### 3. Running at servers

After you put the code on your servers by ssh, and at your computer, please run the bellow code:

`ssh -L 8888:127.0.0.1:8000 username@serveraddress.com`

Next step, on your computer browser, input the address
http://localhost:8888/demo , the page look like:

<img src="./readme_img/upload_page.png" width="400" hegiht="600" align=center />

You can upload your video, and click 'Calculate' button to run this model. When the running is finish, the result page look like:

<img src="./readme_img/result_sum_page.png" width="400" hegiht="600" align=center />

There are each detected face and result video in the result summary page, and you can click each face go to the face stat page, look like:

<img src="./readme_img/face_stat_page.png" width="400" hegiht="600" align=center />

There are this face detected time, and emotion data summary chart at that, and each frame detail data you can download from your servers file path is `./static/result/data/`. 

#### 4. In the own computer

If you want to run the application at your own computer, you need get your video path and run this code:

`./demo_for_local.sh your_video_path`

The result file path is `./static/result/`

#### 5. Environment

I develop this application in the python3 at ubuntu, so the python3 command is `python`, if your python3 command is `python3`, please change `demo_for_local.sh` file.

#### 6. Problem (Need Improved)

1. Improve the FPS.

   Now, the face detection model is MTCNN, and face comparison model is FaceNet, both from `face_recognition`. A new and light model by developer is better.

2. Using combined model.

   Now there are many stages, I believe that a combined model that combining many previous stages is better.

3. Web Application suitable for every condition.

   Now, the web application (front-end and back-end) used a localhost API connect (localhost:8888/xxxx), so if you change your address, many static material would not appear, so I think using a link that can be used in everywhere is better.

