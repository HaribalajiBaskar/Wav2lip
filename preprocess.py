import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

from yoloface.face_detector import YoloDetector

parser = argparse.ArgumentParser()


def face_detect(image):
    model = YoloDetector(device='cpu:0')
    valid_images = []
    ind = 0

    bboxes, points = model.predict(image)
    if len(bboxes[0]) == 0:
        print("Faulty Image Found")
        ind += 1
    elif not np.count_nonzero(image):
        print("Faulty Image Found Black")
        ind += 1
    else:
        valid_images.append(ind)
        x1, y1, x2, y2 = bboxes[0][0]
        face_image = image[y1:y2, x1:x2]

    del model
    return face_image


parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

args = parser.parse_args()

fa = [face_detect]

template = 'ffmpeg -loglevel panic -y -i "{}" -strict -2 "{}"'

# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
resize_factor = 1
rotate = False
corp = [0, -1, 0, -1]


def process_video_file(vfile, args, gpu_id):
    print(vfile)
    video_stream = cv2.VideoCapture(vfile)

    vidname = os.path.basename(vfile).split('.')[0]
    print("fulldir", vidname)
    dirname = vfile.split('\\')[-2]
    print("fulldir", dirname)

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    print("fulldir", fulldir)
    os.makedirs(fulldir, exist_ok=True)

    frames = []
    i = -1
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

        preds = [face_detect(frame)]

        for image in preds:
            i += 1
            cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), image)


def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('.')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    print(command)
    subprocess.call(command, shell=True)


def mp_handler(job):
    vfile, args, gpu_id = job
    try:
        process_video_file(vfile, args, gpu_id)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main(args):
    print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

    filelist = glob(path.join(args.data_root, '*.mp4'))

    jobs = [(vfile, args, i % args.ngpu) for i, vfile in enumerate(filelist)]
    p = ThreadPoolExecutor(args.ngpu)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

    print('Dumping audios...')

    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main(args)
