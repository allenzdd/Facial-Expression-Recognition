from django.shortcuts import render
from .forms import UploadFileForm
from face_expression.models import UploadVideos
from src.for_web.calculate_time import calculate_video_time
from src.model.face_recognition.demo import main as FR

import os



# Create your views here.
def index(request):
    return render(request, 'index.html')

def chart(request):
    return render(request, 'charts.html')

def demo(request):
    try:
        cmd = 'rm static/result/video/output.webm'
        cmd_person = 'rm -rf static/result/person/*'
        cmd_videos = 'rm -rf static/media/*.mp4'
        os.system(cmd)
        os.system(cmd_person)
        os.system(cmd_videos)
    except:
        pass
    try:
        videos = UploadVideos()
        lst = videos.objects.all()
        lst.delete()
    except:
        pass
    return render(request, 'demo.html')

def upload_progress(request):
    if request.method == 'POST':
        videos = UploadVideos()

        temp = request.FILES.get('upload')

        videos.video = temp
        videos.save()

        FR("static/media/" + str(temp))
        # convert avi to webm
        cmd = "ffmpeg -i static/result/video/output.avi -cpu-used 8 -threads 16  static/result/video/output.webm"
        os.system(cmd)

        return render(request, 'charts.html')

    else:
        form = UploadFileForm()
    return render(request, 'charts.html', {'form': form})

def face_stat(request):
    face_id = request.GET['face_id']
    data_url = "static/result/data/summary.csv"
    time_lst = calculate_video_time(data_url, int(face_id))
    return render(request, 'face_stat.html', {"face_id": face_id, "time_lst": time_lst})

