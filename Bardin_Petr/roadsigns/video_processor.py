import cv2


class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector

    def run(self, stream, callback, output_video_file=None, show_rt_window=False):
        output = None

        while stream.isOpened():
            _, frame = stream.read()
            if output_video_file and output is None:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                output = cv2.VideoWriter(
                    output_video_file, fourcc, stream.get(cv2.CAP_PROP_FPS), frame.shape[1::-1])
            try:
                img, data = self.detector(frame)
                callback(int(stream.get(cv2.CAP_PROP_POS_MSEC)), img, data)
            except Exception as ex:
                print(ex)
                break

            if output:
                output.write(img)

            if show_rt_window:
                cv2.imshow('Processed', img)
                if cv2.waitKey(1) == ord('q'):
                    break

        stream.release()
        if output:
            output.release()
        cv2.destroyAllWindows()
