#from PIL import Image
#import numpy as np
import os
import requests

def download_video(video_url, save_path):
    r = requests.get(video_url)
    with open(save_path, 'wb') as f:
        f.write(r.content)

def batch_download():
    file_path = r'F:\download_web_video_tmp\2_month.txt'
    with open (file_path, 'r') as f:
        file_list = f.readlines()

    for i, file in enumerate(file_list):
        real_dir, error_dir, file_url = file.split()
        tmp_real_dir = os.path.join(r'F:\download_web_video_tmp', real_dir)
        file_dir = os.path.join(tmp_real_dir, error_dir)

        if not os.path.exists(tmp_real_dir):
            os.mkdir(tmp_real_dir)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        file_save_path = os.path.join(file_dir, os.path.basename(file_url))

        try:
            download_video(file_url, file_save_path)
        except:
            print("download error:", i)
        if i % 100 == 0:
            print("finished:", i)

def analysis_cgwx_video(url):
    from bs4 import BeautifulSoup

    response = requests.get(url)
    html = response.text

    print(response)
    print(html)

    soup = BeautifulSoup(html, "html.parser")

    # Find all images
    images = soup.find_all("video")
    print(images)
    # Download images
    for image in images:
        image_url = image["src"]
        response = requests.get(image_url)
        with open(image_url.split("/")[-1], "wb") as f:
            f.write(response.content)

if __name__ == '__main__':
    url = r'https://www.jl1mall.com/learn/coursePlay?videoId=8&videoList=%5Bobject%20Object%5D&index=0'
    analysis_cgwx_video(url)

