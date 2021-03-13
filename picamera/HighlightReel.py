import subprocess
import shutil
import os
from datetime import datetime
import cv2

class HighlightReel:
    """This class uses the cv2 library to store frames to a managed directory
        and creates an mp4 video of the frames to show highlights from object tracking"""
    def __init__(self, working_dir, reel_folder='highlights'):
        self.pwd = working_dir
        self.reels_path = self.pwd + '/' + reel_folder
        self.current_dir = self._create_reel_dir()

        self.num_frames_saved = 0

    def _create_reel_dir(self):
        now_str = datetime.now().strftime("%m-%d-%YT%H:%M:%S")
        new_dir = self.reels_path + '/' + now_str
        os.mkdir(new_dir)   
        os.mkdir(new_dir + '/tmp')

        return new_dir
    
    def save_frame(self, frame):
        name = f"frame{self.num_frames_saved:04d}.jpg"
        cv2.imwrite(self.current_dir + '/tmp/' + name, frame)     # save frame as JPEG file
        self.num_frames_saved += 1

        return self.current_dir + '/' + name

    def cleanup(self):
        print("Converting captured frames into mp4 highlight video...")
        # Call ffmpeg to turn individual frames into an mp4
        # ffmpeg -framerate 4 -i highlights/01-08-2021T12:08:49/tmp/frame%4d.jpg highlights/01-08-2021T12:08:49/test4.mp4
        subprocess.call(["ffmpeg", "-loglevel", "panic", "-framerate", "4", "-i", f"{self.current_dir}/tmp/frame%4d.jpg", f"{self.current_dir}/highlight.mp4"])
        print("Frames converted successfully, deleting tmp frames")
        # Remove tmp dir which contains the frames used to create the mp4. Mp4 saved in current_dir
        shutil.rmtree(f"{self.current_dir}/tmp/")
        print("Highlight reel cleanup success")