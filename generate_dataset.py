import argparse
import data_creator

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='generate_dataset.py')
    parser.add_argument('--download_ttfs', type=bool, default=False, help='whether to download ttfs')
    parser.add_argument('--create_imgs', type=bool, default=False, help='whether to create image files')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    if args.download_ttfs:
        data_creator.get_font_ttfs()
    if args.create_imgs:
        data_creator.create_alphabet_for_each_ttf()
    data_creator.create_dataset()
    print('dataset .jsonl files written to project root')