#-----------------------------------------------------------------------#
#   predict_video.py tracks a single user-selected target through a recorded
#   video, at the model's native input resolution.
#
#   Workflow:
#     1. Position an exact 960x960 (the model's input size) crop box on the
#        first frame with the mouse -- this avoids shrinking the frame down
#        to fit the model input, which hurts detection of small targets.
#     2. Inside that crop, drag a rectangle around the target you care about.
#     3. Optionally zero out everything else in the crop.
#     4. Optionally give the target's real-world width to get mm displacement.
#     5. Optionally track with Deep SORT, so the target keeps its identity
#        through brief occlusion or if it temporarily leaves/re-enters the ROI
#        (default: simpler per-frame "box fully inside the ROI" selection).
#
#   Usage:
#       python predict_video.py "./videos/2026-07-21 145954.mov"
#       python predict_video.py                       # uses the default path below
#       python predict_video.py my_video.mp4 --use-deepsort --step 2
#-----------------------------------------------------------------------#
import argparse
import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from predict import addMask
from plot_template import plot_template
from utils_ds.draw import draw_boxes
from utils_ds.parser import get_config
from utils.utils import xyxy2xywh
from nets import build_tracker

DEFAULT_VIDEO = './videos/2026-07-21 145954.mov'


def select_fixed_crop(frame_bgr, crop_size):
    """Let the user position (not resize) an exact crop_size box with the mouse."""
    h, w = frame_bgr.shape[:2]
    cw, ch = min(crop_size[0], w), min(crop_size[1], h)

    state = {'cx': w // 2, 'cy': h // 2, 'clicked': False}

    def on_mouse(event, x, y, flags, param):
        state['cx'], state['cy'] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            state['clicked'] = True

    window = f'Position the {cw}x{ch} crop box - left-click to confirm (ESC to accept current spot)'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print(f'Move the mouse to position the {cw}x{ch} crop box over the region you want to detect in,')
    print('then left-click to confirm (or press ESC to accept the current position).')

    while not state['clicked']:
        x1 = min(max(state['cx'] - cw // 2, 0), w - cw)
        y1 = min(max(state['cy'] - ch // 2, 0), h - ch)
        preview = frame_bgr.copy()
        cv2.rectangle(preview, (x1, y1), (x1 + cw, y1 + ch), (0, 255, 255), 2)
        cv2.imshow(window, preview)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    x1 = min(max(state['cx'] - cw // 2, 0), w - cw)
    y1 = min(max(state['cy'] - ch // 2, 0), h - ch)
    cv2.destroyWindow(window)
    return [x1, y1, x1 + cw, y1 + ch]


def select_roi(frame_bgr, max_display_height=900):
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_display_height / h)
    display = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr

    window = 'Select target ROI - drag a box, then press ENTER/SPACE (ESC to retry)'
    while True:
        print('Drag a rectangle around the target you want to track, a little generous around its edges,')
        print('then press ENTER or SPACE to confirm (ESC to retry with an empty selection).')
        x, y, bw, bh = map(int, cv2.selectROI(window, display, showCrosshair=True, fromCenter=False))
        cv2.destroyWindow(window)
        if bw > 0 and bh > 0:
            return [int(round(x / scale)), int(round(y / scale)),
                     int(round((x + bw) / scale)), int(round((y + bh) / scale))]
        print('Empty selection, please try again.\n')


def ask_yes_no(prompt, default=False):
    suffix = ' [Y/n] ' if default else ' [y/N] '
    while True:
        answer = input(prompt + suffix).strip().lower()
        if not answer:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print('Please answer y or n.')


def ask_positive_float(prompt):
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            print('Please enter a number (or leave blank to skip).')
            continue
        if value <= 0:
            print('Please enter a positive number (or leave blank to skip).')
            continue
        return value


def box_in_roi(box, roi):
    x1, y1, x2, y2 = box[:4]
    rx1, ry1, rx2, ry2 = roi
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2


def pick_target_box(results, roi):
    """Among all boxes YOLO found this frame, keep the highest-confidence one
    that lies completely inside the user-selected ROI. No identity persists
    across frames -- each frame is judged independently."""
    if results is None:
        return None
    candidates = [b for b in results[0].cpu().numpy() if box_in_roi(b, roi)]
    if not candidates:
        return None
    candidates.sort(key=lambda b: b[4], reverse=True)
    return candidates[0]


def update_deepsort_target(outputs, roi, target_id):
    """outputs: Deep SORT's (#tracks, 5) array of [x1,y1,x2,y2,track_id], or empty.
    Follows target_id if it's still being tracked; otherwise (first frame, or
    after the target was lost and reappeared under a new id) locks onto
    whichever confirmed track is fully inside the ROI."""
    if outputs is None or len(outputs) == 0:
        return None, target_id

    outputs = np.asarray(outputs)
    if target_id is not None:
        match = outputs[outputs[:, 4] == target_id]
        if len(match):
            return match[0][:4], target_id

    candidates = [row for row in outputs if box_in_roi(row, roi)]
    if candidates:
        chosen = candidates[0]
        return chosen[:4], int(chosen[4])
    return None, target_id


def main():
    parser = argparse.ArgumentParser(description='Track a single user-selected target through a recorded video.')
    parser.add_argument('video', nargs='?', default=DEFAULT_VIDEO, help='Path to the recorded video')
    parser.add_argument('--out-root', default='./detections', help='Root folder for outputs')
    parser.add_argument('--step', type=int, default=1, help='Process every Nth frame (default: every frame)')
    parser.add_argument('--start', type=int, default=0, help='First frame index to process')
    parser.add_argument('--end', type=int, default=None, help='Last frame index (exclusive) to process, default: end of video')
    parser.add_argument('--save-all-frames', action='store_true',
                         help='Also save frames where the target was not detected (default: only save successful detections)')
    parser.add_argument('--use-deepsort', action='store_true',
                         help='Track with Deep SORT instead of per-frame ROI containment (skips the interactive prompt)')
    parser.add_argument('--config-deepsort', default='./model_data/deep_sort.yaml', help='Deep SORT config file')
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f'Video not found: {args.video}')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = args.end if args.end is not None else (frame_count if frame_count > 0 else 10**9)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError('Could not read the first frame of the video.')

    print(f'Video: {args.video}')
    print(f'{first_frame.shape[1]}x{first_frame.shape[0]} @ {fps:.2f} fps, {frame_count} frames')

    yolo = YOLO()
    crop_x1, crop_y1, crop_x2, crop_y2 = select_fixed_crop(
        first_frame, crop_size=(yolo.input_shape[1], yolo.input_shape[0]))
    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
    print(f'Crop box in the original frame (x1,y1,x2,y2): {[crop_x1, crop_y1, crop_x2, crop_y2]}')

    first_crop = first_frame[crop_y1:crop_y2, crop_x1:crop_x2]
    roi = select_roi(first_crop)
    print(f'Selected ROI within the crop (x1,y1,x2,y2): {roi}')

    mask_background = ask_yes_no('Zero out everything outside the selected area?', default=False)
    target_width_cm = ask_positive_float(
        'Real-world width of the target in cm, matching its detected box width '
        '(leave blank to skip mm conversion): '
    )
    use_deepsort = args.use_deepsort or ask_yes_no(
        'Use Deep SORT to keep following the target through brief occlusion or re-entry?', default=False)

    deepsort = None
    target_id = None
    if use_deepsort:
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)
        deepsort = build_tracker(cfg, use_cuda=torch.cuda.is_available())

    video_stem = os.path.splitext(os.path.basename(args.video))[0].replace(' ', '_')
    frames_dir = os.path.join(args.out_root, 'frames', video_stem)
    videos_dir = os.path.join(args.out_root, 'videos')
    motion_dir = os.path.join(args.out_root, 'motion', video_stem)
    for d in (frames_dir, videos_dir, motion_dir):
        os.makedirs(d, exist_ok=True)

    out_video_path = os.path.join(videos_dir, video_stem + '_detected.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps / max(args.step, 1), (crop_w, crop_h))

    rx1, ry1, rx2, ry2 = roi
    records = []  # frame_idx, time_s, cx_px, cy_px, w_px, h_px, conf, track_id

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx = args.start
    pbar = tqdm(total=max(end_frame - args.start, 0) if end_frame < 10**9 else None, desc='Tracking')
    while frame_idx < end_frame:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if (frame_idx - args.start) % args.step == 0:
            crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            if mask_background:
                pil_image = addMask(pil_image, roi, crop=False)

            yolo.results = None
            yolo.detect_image(pil_image)

            conf_value = None
            if use_deepsort:
                if yolo.results is not None:
                    det = yolo.results[0]
                    bbox_xywh = xyxy2xywh(det[:, :4])
                    confs = det[:, 4:5]
                    outputs = deepsort.update(bbox_xywh, confs, crop_bgr)
                else:
                    outputs = np.empty((0, 5))
                box, target_id = update_deepsort_target(outputs, roi, target_id)
            else:
                box = pick_target_box(yolo.results, roi)

            annotated = crop_bgr.copy()
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

            if box is not None:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if not use_deepsort:
                    conf_value = box[4]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w_px, h_px = x2 - x1, y2 - y1
                records.append((frame_idx, frame_idx / fps, cx, cy, w_px, h_px,
                                conf_value if conf_value is not None else np.nan,
                                target_id if target_id is not None else np.nan))

                identity = target_id if target_id is not None else 0
                annotated = draw_boxes(annotated, [[x1, y1, x2, y2]], identities=[identity])
                cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                label = f'frame {frame_idx}'
                if conf_value is not None:
                    label += f'  conf {conf_value:.2f}'
                if target_id is not None:
                    label += f'  id {target_id}'
                cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_idx:06d}.jpg'), annotated)
            else:
                cv2.putText(annotated, f'frame {frame_idx}  target not detected', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if args.save_all_frames:
                    cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_idx:06d}.jpg'), annotated)

            video_writer.write(annotated)

        frame_idx += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    video_writer.release()

    if not records:
        print('No detections of the selected target were found in the video.')
        return

    records = np.array(records, dtype=float)
    frame_idxs, times, cx, cy, w_px, h_px, conf, track_id = records.T

    dx_px, dy_px = cx - cx[0], cy - cy[0]

    mm_per_px = (target_width_cm * 10.0) / np.nanmean(w_px) if target_width_cm else None
    dx_mm = dx_px * mm_per_px if mm_per_px else None
    dy_mm = dy_px * mm_per_px if mm_per_px else None

    def fmt(v):
        return '' if np.isnan(v) else v

    motion_csv = os.path.join(motion_dir, 'motion.csv')
    with open(motion_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        header = ['frame', 'time_s', 'center_x_px', 'center_y_px', 'box_w_px', 'box_h_px',
                  'confidence', 'track_id', 'dx_px', 'dy_px']
        if mm_per_px is not None:
            header += ['dx_mm', 'dy_mm']
        csv_writer.writerow(header)
        for i in range(len(records)):
            row = [int(frame_idxs[i]), times[i], cx[i], cy[i], w_px[i], h_px[i],
                   fmt(conf[i]), fmt(track_id[i]), dx_px[i], dy_px[i]]
            if mm_per_px is not None:
                row += [dx_mm[i], dy_mm[i]]
            csv_writer.writerow(row)

    with open(os.path.join(motion_dir, 'config.txt'), 'w') as f:
        f.write(f'video: {args.video}\n')
        f.write(f'crop box in original frame (x1,y1,x2,y2): {[crop_x1, crop_y1, crop_x2, crop_y2]}\n')
        f.write(f'roi within crop (x1,y1,x2,y2): {roi}\n')
        f.write(f'mask_background: {mask_background}\n')
        f.write(f'use_deepsort: {use_deepsort}\n')
        f.write(f'target_width_cm: {target_width_cm}\n')
        f.write(f'mean detected box width (px): {np.nanmean(w_px):.3f}\n')
        if mm_per_px is not None:
            f.write(f'mm per pixel: {mm_per_px:.5f}\n')
        f.write(f'frames with detection: {len(records)}\n')

    print(f'Detected the target in {len(records)} frames.')
    print(f'Motion data saved to {motion_csv}')
    print(f'Annotated video saved to {out_video_path}')
    print(f'Annotated frames saved to {frames_dir}')

    print('A plot window will open next -- close it to finish.')
    if mm_per_px is not None:
        plot_template([times, dx_mm], [times, dy_mm],
                      legend_labels=['Horizontal (x)', 'Vertical (y)'],
                      xlabel='Time (s)', ylabel='Displacement (mm)',
                      save_path=os.path.join(motion_dir, 'displacement_mm.svg'))
    else:
        plot_template([times, dx_px], [times, dy_px],
                      legend_labels=['Horizontal (x)', 'Vertical (y)'],
                      xlabel='Time (s)', ylabel='Displacement (px)',
                      save_path=os.path.join(motion_dir, 'displacement_px.svg'))


if __name__ == '__main__':
    main()
