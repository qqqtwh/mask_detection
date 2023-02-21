
import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
        if yolo.output !='':
            r_image.save(yolo.output)


FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--model', type=str,help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
    parser.add_argument('--anchors', type=str,help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
    parser.add_argument('--classes', type=str,help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
    parser.add_argument('--gpu_num', type=int,help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))

    parser.add_argument('--image', action="store_true",help='Image detection mode, will ignore all positional arguments')
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,
        help = "Video input path"
    )
    parser.add_argument("--output", nargs='?', type=str, default="",help = "[Optional] Video output path")

    FLAGS = parser.parse_args()

    if hasattr(FLAGS,'image'):
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
