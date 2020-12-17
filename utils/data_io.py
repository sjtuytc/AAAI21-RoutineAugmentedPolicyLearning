import os
import json
import ffmpeg
import pickle
import sys
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc, resize
import numpy as np
import cv2


def imgseq2video(imgseq, name="pick_up", decode="mp4v", folder=None, fps=3, o_h=500, o_w=500,
                 full_path=None, rgb_to_bgr=True, verbose=True):
    """
    Generate a video from a img sequence list.
    :param imgseq: RGB image frames.
    :param name: video file name.
    :param decode: video decoder type, X264 is not working.
    :param folder: saved to which folder.
    :param fps: fps of saved video.
    :param o_h: height of video.
    :param o_w: width of video
    :param full_path: full path to the video, if not None, overwrite folder and name.
    :param rgb_to_bgr: convert rgb image to bgr img.
    :param verbose: whether to print save path.
    :return: None.
    """
    if len(imgseq) < 1:
        print("[WARNING] Try to save empty video.")
        return
    # Suppress OpenCV and ffmpeg output.
    sys.stdout = open(os.devnull, "w")
    if full_path is not None:
        assert ".mp4" in full_path[-4:], "Full path should end with .mp4"
        tmp_path = full_path[:-4] + "tmp" + ".mp4"
        path = full_path
    else:
        tmp_path = name + "tmp.mp4" if folder is None else os.path.join(folder, name + "tmp.mp4")
        path = name + ".mp4" if folder is None else os.path.join(folder, name + ".mp4")
    fourcc = VideoWriter_fourcc(*decode)
    videoWriter = VideoWriter(tmp_path, fourcc, fps, (o_w, o_h))
    for img in imgseq:
        img = np.uint8(img)
        if img.shape[0] == 3:
            # needs to be in shape of oh, ow, 3
            img = img.transpose(1, 2, 0)
        if rgb_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize(img, (o_w, o_h))
        videoWriter.write(img)
    videoWriter.release()
    (
        ffmpeg
            .input(tmp_path)
            .output(path, vcodec="h264", loglevel="error")
            .overwrite_output()
            .run()
    )
    # .global_args('-loglevel', 'error')
    # print("should be blocked")
    os.remove(tmp_path)
    sys.stdout = sys.__stdout__
    if verbose:
        print("Video saved to", path, "with ", len(imgseq), " total frames.")
    return path


def save_into_json(save_obj, folder, file_name="test", full_path=None, verbose=True):
    if full_path is None:
        full_path = os.path.join(folder, str(file_name) + ".json")
    gt_file = open(full_path, 'w', encoding='utf-8')
    json.dump(save_obj, gt_file)
    if verbose:
        print("Current obj saved at", full_path)
    gt_file.close()
    return full_path


def read_from_json(folder, file_name="test", full_path=None, verbose=False):
    if full_path is None:
        full_path = os.path.join(folder, str(file_name) + ".json")
    file_obj = open(full_path)
    data_obj = json.load(file_obj)
    file_obj.close()
    if verbose:
        print("Read obj from", full_path)
    return data_obj


def save_into_img(img_matrix, folder=None, img_name=None, verbose=False):
    full_path = os.path.join(folder, img_name + ".jpg")
    plt.imsave(full_path, img_matrix, dpi=1000)
    if verbose:
        print("Cur img saved at", os.path.join(full_path))
    return full_path


def save_into_pkl(save_obj, full_path=None, name="test", folder="", verbose=False):
    if full_path is None:
        full_path = os.path.join(folder, str(name) + '.pkl')
    output = open(full_path, 'wb')
    pickle.dump(save_obj, output)
    output.close()
    if verbose:
        print("Current obj saved at", os.path.join(full_path))
    return full_path


def read_from_pkl(name="test", folder="", full_path=None):
    if full_path is None:
        full_path = os.path.join(folder, str(name) + ".pkl")
    pkl_file = open(full_path, 'rb')
    return_obj = pickle.load(pkl_file)
    pkl_file.close()
    return return_obj


def load_routine_action(exp_folder, file_name, routine_num, routine_ablation):
    result_data = read_from_json(folder=exp_folder, file_name=file_name, verbose=True)
    routine_key = "routines"
    if routine_ablation != "":
        routine_key = routine_ablation
    return result_data[routine_key][:routine_num]


def save_routine_action(result_data, exp_folder):
    save_into_json(result_data, file_name="routine_library", folder=exp_folder)
