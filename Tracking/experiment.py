"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5
import ps5_utils

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))

def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    save_frames = {
        29: os.path.join(output_dir, 'ps5-5-a-1.png'),
        56: os.path.join(output_dir, 'ps5-5-a-2.png'),
        71: os.path.join(output_dir, 'ps5-5-a-3.png')
    }

    imgs_dir = os.path.join(input_dir, "TUDCampus")
    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    
    imgs_list.sort()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    max_track_age = 2

    frame_num = 1

    id_counter = 0
    kalman_objects = {}
    for image_path in imgs_list:
        frame = cv2.imread(os.path.join(imgs_dir, image_path))
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        
        # for i in range(len(rects)):
        #     x, y, w, h = rects[i]
        #     associated = False
        #     for k, v in kalman_objects.items():
        #         box1 = (v.state[0], v.state[1], v.width, v.height)
        #         box2 = (x, y, w, h)
        #         iou = ps5_utils.calculate_iou(box1, box2)
        #         if iou > 0.5:
        #             kalman_objects[k].process(x, y)
        #             associated = True
        #             break
        #     if not associated:
        #         kalman_objects[id_counter] = ps5.KalmanFilter(x, y)
        #         kalman_objects[id_counter].width = w
        #         kalman_objects[id_counter].height = h
        #         id_counter += 1
        
        # keys_to_delete = []
        # for i in kalman_objects:
        #     for j in kalman_objects:
        #         if i == j:
        #             continue
        #         box1 = (kalman_objects[i].state[0], kalman_objects[i].state[1], kalman_objects[i].width, kalman_objects[i].height)
        #         box2 = (kalman_objects[j].state[0], kalman_objects[j].state[1], kalman_objects[j].width, kalman_objects[j].height)
        #         iou = ps5_utils.calculate_iou(box1, box2)
        #         if iou > 0.4:
        #             print("lol")
        #             # Determine which bounding box is bigger
        #             if kalman_objects[i].width * kalman_objects[i].height < kalman_objects[j].width * kalman_objects[j].height:
        #                 keys_to_delete.append(i)
        #                 break
        #             else:
        #                 keys_to_delete.append(j)

        # # Delete identified keys
        # for key in keys_to_delete:
        #     if key in kalman_objects:
        #         del kalman_objects[key]

        keys_to_delete = []
        for key, obj in kalman_objects.items():
            obj.age += 1
            if obj.age > max_track_age:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del kalman_objects[key]

        for i in range(len(rects)):
            x, y, w, h = rects[i]
            associated = False
            for k, v in kalman_objects.items():
                box1 = (v.state[0], v.state[1], v.width, v.height)
                box2 = (x, y, w, h)
                iou = ps5_utils.calculate_iou(box1, box2)
                if iou > 0.5:
                    v.process(x, y)
                    v.age = 0  # Reset age
                    associated = True
                    break
            if not associated:
                kalman_objects[id_counter] = ps5.KalmanFilter(x, y)
                kalman_objects[id_counter].width = w
                kalman_objects[id_counter].height = h
                kalman_objects[id_counter].age = 0
                id_counter += 1
        

        for i, obj in kalman_objects.items():
            object_x = int(obj.state[0])
            object_y = int(obj.state[1])
            cv2.putText(frame, str(i), (object_x, object_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (object_x, object_y), (object_x + obj.width, object_y + obj.height), (0, 255, 0), 2)
        
        print("Frame: ", frame_num)
        # Render and save output, if indicated
        if frame_num in save_frames:
            cv2.imwrite(save_frames[frame_num], frame)
        frame_num += 1


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    template_rect = {'x': 88, 'y': 36, 'w': 35, 'h': 170}

    save_frames = {
        60 :  os.path.join(output_dir,   'ps5-6-a-1.png'),
        160: os.path.join(output_dir,   'ps5-6-a-2.png'),
        186: os.path.join(output_dir,   'ps5-6-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_6(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "follow"))


if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    # part_5()
    part_6()
