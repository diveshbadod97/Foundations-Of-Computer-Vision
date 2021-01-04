import time
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from networkx.drawing.tests.test_pylab import plt


class LaneDetectionClass:

    def hls2rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

    def rgb2bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def bgr2rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def rgb2hls(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def compute_white_yellow(self, hls_img):
        img_hls_yellow_bin = np.zeros_like(hls_img[:, :, 0])
        img_hls_yellow_bin[((hls_img[:, :, 0] >= 15) & (hls_img[:, :, 0] <= 35))
                           & ((hls_img[:, :, 1] >= 30) & (hls_img[:, :, 1] <= 204))
                           & ((hls_img[:, :, 2] >= 115) & (hls_img[:, :, 2] <= 255))
                           ] = 1
        img_hls_white_bin = np.zeros_like(hls_img[:, :, 0])
        img_hls_white_bin[((hls_img[:, :, 0] >= 0) & (hls_img[:, :, 0] <= 255))
                          & ((hls_img[:, :, 1] >= 200) & (hls_img[:, :, 1] <= 255))
                          & ((hls_img[:, :, 2] >= 0) & (hls_img[:, :, 2] <= 255))
                          ] = 1
        img_hls_white_yellow_bin = np.zeros_like(hls_img[:, :, 0])
        img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1
        return img_hls_white_yellow_bin

    def blur(self, bin_img):
        kernel_size = 11

        return cv2.GaussianBlur(bin_img, (kernel_size, kernel_size), 0)

    def onlyLanes(self, input_img):
        rows, cols = input_img.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.55]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.55]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        mask = np.zeros_like(input_img)
        cv2.fillPoly(mask, vertices, 255)
        input_img = cv2.bitwise_and(input_img, mask)

        return input_img

    def hough_lines(self, img):
        return cv2.HoughLinesP(img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=50, maxLineGap=100)

    def draw_lines(self, image, lines):

        if not lines is None:

            for line in lines:

                for x1, y1, x2, y2 in line:
                    cv2.line(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    # cv2.circle(image, center=(x2, y2), radius=5, color=(255, 0, 0), thickness=-1)
                    # cv2.circle(image, center=(x3, y3), radius=5, color=(255, 0, 0), thickness=-1)

                    # If slope is negative, then it is left lane line
                    # cv2.line(image, center=(x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
                    # cv2.circle(image, center=(x2, y2), radius=5, color=(0, 0, 255), thickness=-1)
                    # cv2.circle(image, center=(x3, y3), radius=5, color=(0, 0, 255), thickness=-1)
        return image

    def detect_cars(self, image):
        car_cascade = cv2.CascadeClassifier('cars1.xml')
        cars = car_cascade.detectMultiScale(image, 1.3, 2, minSize=(70, 70), maxSize=(200, 200))
        for (x, y, w, h) in cars:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image

    def process(self, image):
        image = self.bgr2rgb(image)
        img_hls = self.rgb2hls(np.uint8(image))
        white_yellow = self.compute_white_yellow(img_hls)
        regions = self.onlyLanes(white_yellow)
        blurred = self.blur(regions)
        lines = self.hough_lines(blurred)
        temp = self.draw_lines(image, lines)
        #output = self.detect_cars(temp)
        # plt.figure()
        # plt.imshow(img_hls)
        # plt.savefig('A.png', bbox_inches='tight')
        # plt.figure()
        # plt.imshow(white_yellow)
        # plt.savefig('B.png', bbox_inches='tight')
        # plt.figure()
        # plt.imshow(regions)
        # plt.savefig('C.png', bbox_inches='tight')
        # plt.figure()
        # plt.imshow(blurred)
        # plt.savefig('D.png', bbox_inches='tight')
        # plt.figure()
        # plt.imshow(lines)
        # plt.savefig('E.png', bbox_inches='tight')
        # plt.figure()
        # plt.imshow(temp)
        # plt.savefig('F.png', bbox_inches='tight')
        # plt.show()
        # plt.imshow(output)
        # plt.savefig('G.png', bbox_inches='tight')
        # plt.show()
        return temp


def process_video(video_input):
    inpt = video_input + '.mp4'
    otpt = video_input + 'out.avi'
    detector = LaneDetectionClass()
    cap = cv2.VideoCapture(inpt)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(otpt, fourcc, 20, (frame_width, frame_height))
    cnt = 0
    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            start = time.time()
            if cnt == 0:
                etrm_strt = start
            img = detector.bgr2rgb(image)
            output_image = detector.process(img)
            #cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
            end = time.time()
            out.write(detector.rgb2bgr(output_image))
            cnt += 1
        else:
            etrm_end = time.time()
            print('Average Time : ' + str((etrm_end - etrm_strt) / cnt) + 's')
            break
    cap.release()
    #out.release()
    return cnt


if __name__ == '__main__':
    start = time.time()
    #cnt = process_video('C')
    # stop the timer
    end = time.time()
    lc = LaneDetectionClass()
    # Print the time take
    img = lc.process(cv2.imread("00000.jpg"))
    cv2.imshow("out", img)
    cv2.waitKey(0)
    #print("Total : " + str((end - start)) + "s for " + str(cnt) + "frames")
