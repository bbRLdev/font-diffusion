import argparse
import data_creator

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='generate_dataset.py')
    parser.add_argument("--create_imgs", action="store_true", help="whether to download ttfs")
    parser.add_argument("--download_ttfs", action="store_true", help='whether to create image files')
    parser.add_argument('--model', type=str, default='CLIP', help='whether to create image files')
    parser.add_argument('--image_size', type=int, default=512, help='size of images to generate')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    if args.download_ttfs:
        data_creator.get_font_ttfs()
    if args.create_imgs:
        data_creator.create_alphabet_for_each_ttf(img_size=args.image_size)
    if args.model == 'ViT':
        data_creator.create_dataset(f'font-images-{args.image_size}', for_vit=True)
    elif args.model == 'SD':
        data_creator.create_dataset_for_sd(f'font-images-{args.image_size}')
    else:
        data_creator.create_dataset(f'font-images-{args.image_size}', for_vit=False)

    print('dataset .jsonl files written to project root')