import argparse


argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",help="Configuration file to load the train and validation data.")

    args = vars(parser.parse_args())
    return args


def load_data():
    pass


def main():
    args = parse_args()



if __name__ == "__main__":
    main()
