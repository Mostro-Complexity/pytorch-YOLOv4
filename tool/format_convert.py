# import package
import labelme2coco
import labelme2yolo
import argparse


def main():
    parser = argparse.ArgumentParser(description='Annotation Convertor')
    parser.add_argument('-t', '--desired_type', type=str, choices=['coco', 'yolo'])
    parser.add_argument('-i', '--input_dir', type=str, default='data/labelme')
    parser.add_argument('-o', '--output_path', type=str, required=True)
    args = parser.parse_args()

    if args.desired_type == 'coco':
        # convert labelme annotations to coco
        labelme2coco.convert(args.input_dir, args.output_path)
    elif args.desired_type == 'yolo':
        labelme2yolo.convert(args.input_dir, args.output_path)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
