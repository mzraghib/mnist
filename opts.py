import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', default='input_train', type=str, help='Input file path - train')
    parser.add_argument('--input_validation', default='input_validation', type=str, help='Input file path - validaiton')
    parser.add_argument('--input_test', default='input_test', type=str, help='Input file path - test')

    parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs for training')
    parser.add_argument('--time_steps', default=10, type=int, help='Number of previous time steps to generate future predictions')


    args = parser.parse_args()

    return args
