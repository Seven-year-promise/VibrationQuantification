from Methods.UNet_tf.test import *
from Methods.UNet_tf.util import *
from Methods.Tracking import *
import cv2
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
from Methods.RegionGrowing import Point, RegionGrow
from Methods.QuantificationIndex import QuantificationIndex
import config
DEBUG = False

COLORS = [[133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234],
          [133, 145, 220],
          [50, 145, 100],
          [200, 60, 220],
          [200, 100, 100],
          [50, 220, 20],
          [70, 225, 50],
          [80, 167, 177],
          [177, 145, 234]]

DRAW_FLOW_LINE = False
DRAW_FLOW_POINT = False
SAVE = False
SAVE_VIDEO = False
SHOW = False
SAVE_X_MIN = 0
SAVE_X_MAX = 480
SAVE_Y_MIN = 0
SAVE_Y_MAX = 480

def get_iou(blobA, blobB, ori_shape):
    maskA = np.zeros(ori_shape, np.uint8)
    maskB = np.zeros(ori_shape, np.uint8)
    maskA[blobA] = 1
    maskB[blobB] = 1
    #cv2.imshow("maskA", maskA*255)
    #cv2.imshow("maskB", maskB*255)
    #cv2.waitKey(0)
    AB = np.sum(np.logical_and(maskA, maskB))
    A = np.sum(maskA)
    B = np.sum(maskB)
    #print(A, B, AB)

    return AB / (A + B - AB)

def find_next_point(larva_centers, new_blobs):
    new_center_candidates = []
    for b in new_blobs:
        center = np.round(np.average(b, axis=0))
        new_center_candidates.append([int(center[0]), int(center[1])])

    new_centers = list(map(lambda y: min(new_center_candidates, key=lambda x: ((x[0] - y[0])**2 + (x[1] - y[1])**2 )), larva_centers))

    return new_centers

def larva_tracking(video, model_path):
    unet_test = UNetTestTF()
    unet_test.model.load_graph_frozen(model_path=model_path)
    i = 0
    last_binary, this_binary = None, None

    larva_centers = []
    new_video = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480, 480))

    for im in video:
        i += 1
        unet_test.load_im(im)
        _, this_binary, larva_blobs = unet_test.predict(threshold=0.9, size=44)

        if last_binary is None:
            for b in larva_blobs:
                center = np.round(np.average(b, axis=0))
                larva_centers.append([int(center[0]), int(center[1])])
        else:
            larva_centers = find_next_point(larva_centers, larva_blobs)
        last_binary = this_binary
        tracked_im = im.copy()
        for ct, cr in zip(larva_centers, COLORS[:len(larva_centers)]):
            tracked_im = cv2.circle(tracked_im, center = (ct[1], ct[0]), radius=2, color=cr, thickness=2)
        cv2.imshow("tracked", tracked_im)
        cv2.waitKey(1)
        new_video.append(tracked_im)
        out.write(tracked_im)
    out.release()

class BehaviorQuantify:
    def __init__(self, im_shape, model_path):
        self.unet_test = UNetTestTF()
        self.unet_test.model.load_graph_frozen(model_path=model_path)

        self.video = None
        self.larva_centers = []
        self.larva_percentage_pointss = []
        self.larva_skeletons = []
        self.im_shape = im_shape

        self.needle_tracker = NeedleTracker()
        self.larva_tracker = LarvaTracker()
        self.larva_tracker2 = ParticleFilter(100)

        self.purlse_frame_index = 0
        self.light_area = [50,50,125,125] # x1, y1, x2, y2

        self.RG = RegionGrow()

        self.QutifyIndex = QuantificationIndex(move_thre=25)

    def load_video(self, video):
        self.video = video

    def get_skeleton(self, blob):
        fish_binary = np.zeros(self.im_shape, dtype=np.uint8)
        fish_binary[blob[:, 0], blob[:, 1]] = 1
        skeleton = skeletonize(fish_binary)
        skeleton_cor = np.where(skeleton > 0)
        skeleton_cor = np.array([skeleton_cor[0], skeleton_cor[1]]).reshape(2, -1)
        point1 = skeleton_cor[:, 0]
        point2 = skeleton_cor[:, -1]

        return skeleton_cor
    
    def get_purlse_frame(self):
        if self.video is None:
            print("please load the video first")
        else:
            last_light_hist = 0
            for i, frame in enumerate(self.video):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                light_crop = frame[self.light_area[1]:self.light_area[3], self.light_area[0]:self.light_area[2]]
                light_hist = cv2.calcHist([light_crop], [0], None, [10], [0, 256])
                #cv2.imshow("light crop", light_crop)
                #print(light_hist)
                if i > 0:
                    num_light_points = np.sum(np.abs(light_hist-last_light_hist))

                    if num_light_points > 1500:
                        self.purlse_frame_index = i
                        break
                last_light_hist = light_hist

    def quantification_init(self):
        self.get_purlse_frame()
        self.larva_centers = []
        self.larva_percentage_pointss = []
        self.larva_skeletons = []
        if self.video is None:
            print("please load the video first")
        else:
            self.unet_test.load_im(self.video[self.purlse_frame_index])
            _, _, larva_binary, larva_blobs = self.unet_test.predict(threshold=0.9, size=50)

            # postprocessing
            # morphography
            if DEBUG:
                cv2.imwrite("tracking_saved/original_im.jpg", self.video[self.purlse_frame_index])
                cv2.imwrite("tracking_saved/larva_binary.jpg", larva_binary*255)
            for b in larva_blobs:
                center = np.round(np.average(b, axis=0))
                fish_point = self.unet_test.find_fish_point(fish_mask=larva_binary, fish_blob=b, percentages=[0.1])
                #print(fish_point)
                self.larva_centers.append(center)
                self.larva_percentage_pointss.append(fish_point)
                skel = self.get_skeleton(b)
                self.larva_skeletons.append(skel)
            self.larva_tracker.init_p0(self.larva_percentage_pointss)
            first_gray = self.preprocessing(self.video[self.purlse_frame_index])
            self.larva_tracker2.init_boxes0(first_gray, self.larva_centers, larva_blobs)

    def preprocessing(self, im, strong = False):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if not strong:
            _, (well_x, well_y, _), im_well = well_detection(im, gray)
        else:
            _, (well_x, well_y, _), im_well = well_detection_strong(im, gray, threshold=150)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

        return im_well

    def larva_touched(self, needle_point, n_l_dis_thre):
        """
        decide which larva was touched
        :param needle_point: the needle point of the last frame
        :param n_l_dis_thre: the distance to decide that the larva is touched
        :return: 0, 1, 2, 3: the larva in dex touched, -1: none of the larvae touched
        """
        distances = []
        for c in self.larva_centers:
            d = math.sqrt((c[0] - needle_point[0])**2 + (c[1] - needle_point[1])**2)
            distances.append(d)
        touched_ind = np.argmin(np.array(distances))
        print(distances)
        if distances[touched_ind] < n_l_dis_thre:
            return touched_ind
        else:
            return -1

    def compute_total_distances(self, all_points, ind):
        distance = 0
        #print(all_points)
        for i in range(len(all_points)-1):
            p1 = all_points[i][ind]
            p2 = all_points[i+1][ind]
            d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            distance += d
            #print(p1, p2, d, distance)
        #print(distance)
        return distance

    def local_seg(self, frame, larva_points):
        """
        segmentation for the local area of the larva
        :param frame: frame to generate the local area of the larva
        :param larva_points: the particles output from the tracking procedure
        :return:
            tuned_binary: the optimized binary in original image size
            larva_patches: the larva areas in local size
            fblobs: the larva blobs coordinates in original image size
            tuned_points: the optimized larva centers
        """
        # the fish area should be enlarged to 40 by 40
        fblobs = []
        larva_patches = []
        tuned_points = []
        tuned_binary = frame.copy() #np.zeros(frame.shape, dtype='uint8') #
        tuned_binary[:, :] = 0
        for l_p in larva_points:
            y_ave = int(l_p[0])
            x_ave = int(l_p[1])

            blur = cv2.medianBlur(frame, 5)
            binary = self.RG.regionGrowLocalApply(blur,
                                                  [Point(x_ave, y_ave)], # Point (x, y)
                                                  grad_thre=80,
                                                  binary_high_thre=230,
                                                  binary_low_thre=50,
                                                  size_thre=200)


            ret, labels = cv2.connectedComponents(binary)
            blobs_raw = []
            size = []

            for label in range(1, ret):
                coordinate = np.asarray(np.where(labels == label)).transpose()
                size.append(coordinate.shape[0])
                blobs_raw.append(coordinate)
                tuned_y_ave = np.average(coordinate[:, 0])
                tuned_x_ave = np.average(coordinate[:, 1])
                tuned_points.append([tuned_y_ave, tuned_x_ave])

            biggest_ind = np.argmax(np.array(size))
            blob = blobs_raw[biggest_ind]
            # get the small singe larva patch
            x_min = np.min(blob[:, 1])
            y_min = np.min(blob[:, 0])
            x_max = np.max(blob[:, 1])
            y_max = np.max(blob[:, 0])
            larva_patch = np.zeros(((y_max-y_min+1), (x_max-x_min+1)), np.uint8)
            larva_patch[blob[:, 0] - y_min, blob[:, 1] - x_min] = 1
            larva_patches.append(larva_patch)
            #cv2.imshow("binary", larva_patch + 255)
            #cv2.waitKey(1)
            # remap the flob to the original size

            fblobs.append(blob)
            tuned_binary[blob[:, 0], blob[:, 1]] = 1

        #cv2.imshow("local", tuned_binary*255)
        #cv2.waitKey(1)

        return tuned_binary, larva_patches, fblobs, tuned_points

    def yolo_detect(self):
        # https: // opencv - tutorial.readthedocs.io / en / latest / yolo / yolo.html'
        pass

    def apply_well_mask(self, im, well_mask):
        well_mask = cv2.bitwise_not(well_mask)
        im_copy = im.copy()
        im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)
        im_copy[np.where(well_mask<1)] = 0

        return im_copy

    def quantify(self, save_path, video_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        new_video = []
        new_video2 = []
        new_video3 = []
        new_video4 = []
        new_video5 = []
        old_gray = self.preprocessing(self.video[self.purlse_frame_index])
        well_mask = self.preprocessing(self.video[self.purlse_frame_index], strong=True)
        larva_pointss = []
        larva_patchess = []
        larva_blobss = []
        num_diffss = []

        id_im = 0
        previous = []
        previous.append(old_gray)
        for im in self.video[(self.purlse_frame_index+1):]:
            id_im += 1
            new_gray = self.apply_well_mask(im, well_mask)
            #cv2.imshow("well", new_gray)
            #cv2.waitKey(1)
            im_with_pars = im.copy()
            draw_particles(im_with_pars, self.larva_tracker2.new_particles)
            #larva_points = self.larva_tracker.optical_track(old_gray, new_gray)
            #tracked_im = self.larva_tracker.dense_track(old_im, im)
            #larva_pointss.append(larva_points)
            larva_points, im_diff, num_diffs = self.larva_tracker2.track(previous, new_gray, 15, 0.5)
            tracked_binary, tracked_patches, tracked_blobs, tracked_points = self.local_seg(new_gray, larva_points)
            self.larva_tracker2.resampling_within_blobs(tracked_blobs)
            larva_pointss.append(tracked_points)
            larva_patchess.append(tracked_patches)
            larva_blobss.append(tracked_blobs)
            num_diffss.append(num_diffs)
            """
            modify the larva area according to the rough tracking
            """

            #_ = self.larva_tracker.dense_track(old_im, im)
            #_ = self.larva_tracker.difference_track(previous, new_gray, 10)
            tracked_im = im.copy()
            tracked_im2 = im.copy()
            #for ct, cr in zip(larva_centers, COLORS[:len(larv<a_centers)]):

            if DRAW_FLOW_LINE:
                show_l_points = larva_pointss[::20]
                for i in range(len(show_l_points) - 1):
                    if len(show_l_points[i]) == len(show_l_points[i+1]):
                        for l_p1, l_p2, c in zip(show_l_points[i], show_l_points[i+1], COLORS[:len(show_l_points[i])]):
                            # print(l_p)
                            '''
                            tracked_im = cv2.line(tracked_im, pt1=(int(np.round(l_p1[0])), int(np.round(l_p1[1]))),
                                                  pt2=(int(np.round(l_p2[0])), int(np.round(l_p2[1]))),
                                                  color=c, thickness=2)
                            '''
                            tracked_im = cv2.line(tracked_im, pt1=(int(np.round(l_p1[1])), int(np.round(l_p1[0]))),
                                                  pt2=(int(np.round(l_p2[1])), int(np.round(l_p2[0]))),
                                                  color=c, thickness=3)
                new_video.append(tracked_im)
            #cv2.imshow("tracked_im", tracked_im)
            #cv2.waitKey(10)
            if DRAW_FLOW_POINT:
                for l_p1, c in zip(larva_points, COLORS[:len(larva_points)]):
                    # print(l_p)
                    tracked_im2 = cv2.rectangle(tracked_im2, pt1=(int(np.round(l_p1[1])), int(np.round(l_p1[0]))),
                                          pt2=(int(np.round(l_p1[1]+3)), int(np.round(l_p1[0]+3))),
                                          color=c, thickness=4)
                new_video2.append(tracked_im)
            #cv2.imshow("im_with_pars", im_with_pars)
            #cv2.waitKey(0)
            if SAVE:
                (save_path/video_name).mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(save_path / (video_name + "/ori" + str(id_im) + ".jpg")), im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(str(save_path / (video_name + "/particles_line" + str(id_im) + ".jpg")), tracked_im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(str(save_path / (video_name + "/particles_point" + str(id_im) + ".jpg")), tracked_im2[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(str(save_path / (video_name + "/particles_ori" + str(id_im) + ".jpg")), im_with_pars[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imwrite(str(save_path / (video_name + "/particles_difference" + str(id_im) + ".jpg")), im_diff[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                local_bianary = tracked_binary*255
                cv2.imwrite(str(save_path / (video_name + "/local_seg" + str(id_im) + ".jpg")), local_bianary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                print("saving pictures")
            if SHOW:
                cv2.imshow("local seg", tracked_binary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX]*255)
                cv2.imshow("tracked", tracked_im2[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imshow('new_gray', im_with_pars[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.imshow('diff_im', im_diff[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
                cv2.waitKey(1)
            old_gray = new_gray
            old_im = im
            previous.append(old_gray)
            #print(str(id_im))
            new_video3.append(im_with_pars[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
            new_video4.append(im_diff[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])
            new_video5.append(tracked_binary[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX]*255)
            #print(larva_points[1])


        #print(larva_pointss)
        if SAVE_VIDEO:
            # decide which larva is the one to touch
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_path / (video_name + 'line.avi')), fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))
            #out2 = cv2.VideoWriter(filename=str(save_path / (video_name + 'point.avi')), fourcc=fourcc, fps=20.0, frameSize=(SAVE_Y_MAX-SAVE_Y_MIN, SAVE_X_MAX-SAVE_X_MIN))
            out3 = cv2.VideoWriter(str(save_path / (video_name + 'particles.avi')), fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN, SAVE_X_MAX-SAVE_X_MIN))
            #out4 = cv2.VideoWriter('tracking_saved/output4.avi', fourcc, 20.0, (SAVE_Y_MAX-SAVE_Y_MIN+1, SAVE_X_MAX-SAVE_X_MIN+1))
            out5 = cv2.VideoWriter(str(save_path / (video_name + 'rg.mp4')), mp4, 20.0,
                                   (SAVE_Y_MAX - SAVE_Y_MIN, SAVE_X_MAX - SAVE_X_MIN), False)
            for im, im3, im4, im5 in zip(new_video, new_video3, new_video4, new_video5):
                cv2.imshow("im3", im3)
                cv2.waitKey(10)
                out.write(im)
                out3.write(im3)
                #out4.write(im4)
                out5.write(im5)

            print("saving videos")
            out.release()
            #out2.release()
            out3.release()
            #out4.release()
            out5.release()

        quan_indexes = self.QutifyIndex.get_indexes(larva_first_centers=self.larva_centers,
                                                               larva_pointss=larva_pointss,
                                                               larva_patchess=larva_patchess,
                                                               num_diffss=num_diffss)

        print(quan_indexes)
        return quan_indexes

if __name__ == '__main__':
    behav_quantify = BehaviorQuantify((480, 480), model_path=str(config.UNET_MODEL_PATH))
    base_path = config.QUANTIFY_DATA_PATH
    all_video_paths = base_path.rglob("*.avi")

    #print(all_video_paths)
    with open(config.QUANTIFY_SAVE_PATH / "hts_quantification.csv", "a+", newline='') as f:
        valwriter = csv.writer(f, delimiter=';',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        valwriter.writerow(["compound", "video_path", "t_l", "c_m", "cpt", "t_r", "d_m"])

    for i, v_p in enumerate(all_video_paths):
        print(v_p)
        video = []
        cap = cv2.VideoCapture(str(v_p))
        success, frame = cap.read()
        while success:
            video.append(frame)
            success, frame = cap.read()
        cap.release()

        begin_time = time.clock()
        behav_quantify.load_video(video)
        behav_quantify.quantification_init()
        quan_indexes = behav_quantify.quantify(save_path=config.QUANTIFY_SAVE_PATH, video_name=v_p.name)
        # for saving the quantification
        with open(config.QUANTIFY_SAVE_PATH / "hts_quantification.csv", "a+", newline='') as f:
            valwriter = csv.writer(f, delimiter=';',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for q_i in quan_indexes:
                valwriter.writerow([v_p.parent.name, v_p.name, q_i[0], q_i[1], q_i[2], q_i[3], q_i[4]])
        cv2.destroyAllWindows()

        if i>0:
            break