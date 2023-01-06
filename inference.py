import os
import cv2
import torch
import audio
import platform
import numpy as np
import face_detection
from tqdm import tqdm
from models import Wav2Lip
import matplotlib.pyplot as plt
import json, subprocess, random, string

img_size = 96
resize_factor = 1
rotate = False
full_frames = []
corp = [0, -1, 0, -1]
face_det_batch_size = 16
wav2lip_batch_size = 128
pads = [0, 10, 0, 0]
nosmooth = False
box = [-1, -1, -1, -1]
static = False
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

out_file = "results/result_v3.mp4"


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    valid_images = []
    ind = 0
    pady1, pady2, padx1, padx2 = pads
    updated_images = []
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            print("Faulty Image Found")
            print(ind)
            ind += 1
            continue
            # raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
        elif not np.count_nonzero(image):
            print("Faulty Image Found Black")
            ind += 1
            continue
        else:
            valid_images.append(ind)

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])
        updated_images.append(image)
        ind += 1
    print("Lenght of .........................  ", len(results), len(updated_images))

    boxes = np.array(results)
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(updated_images, boxes)]

    del detector
    return results, valid_images


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if not static:
            face_det_results, valid_images = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results, valid_images = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    updated_frames = []
    #	print(valid_images)
    for ind in valid_images:
        #		print(ind)
        updated_frames.append(frames[ind])

    # frames = updated_frames
    for i, m in enumerate(mels):
        idx = 0 if static else i % len(updated_frames)
        frame_to_save = updated_frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        plt.imshow(face)
        plt.show()

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def inference(checkpoint, face, gaudio):
    if not os.path.isfile(face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face)]
        fps = 25

    else:
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        resize_factor = 1
        rotate = False
        full_frames = []
        corp = [0, -1, 0, -1]
        face_det_batch_size = 16
        wav2lip_batch_size = 128
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = corp
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    if not gaudio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(gaudio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        gaudio = 'temp/temp.wav'

    wav = audio.load_wav(gaudio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    wav_sum = []
    num_parts = len(wav) // len(mel_chunks)
    for i in range(0, len(mel_chunks)):
        wav_sum.append(np.sum(wav[i * num_parts:(i + 1) * num_parts]))

    full_frames = full_frames[:len(mel_chunks)]
    print("........................Length of full frames : ", len(full_frames))
    print("........................Length of mel_chunks : ", len(mel_chunks))
    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)
    j = 0
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(
                                                                        np.ceil(float(len(mel_chunks)) / batch_size)))):
        print(i)
        if i == 0:
            model = load_model(checkpoint)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        plt.imshow(img_batch)
        plt.show()

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            if i == 0:
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
                temp_img = f.copy()
            elif abs(wav_sum[j]) <= 0.15:
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(temp_img)
            else:
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
                temp_img = f.copy()
            j += 1
    out.release()
    # give temp audio path
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(gaudio, 'temp/result.avi', out_file)
    subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
    inference('checkpoints/wav2lip.pth', 'audio/female.mp4','audio/sara_sample.wav')
