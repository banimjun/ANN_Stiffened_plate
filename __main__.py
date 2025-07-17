import argparse
from model import ULSPredict

def parse_args():
    desc = "Tensorflow implementation of ULS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=12, help='The size of batch size')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epoch')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='allcombine.csv',
        help='Excel data directory'
    )
    parser.add_argument(
        '--data_columns',
        type=list,
        default=['tp','beta','hw', 'tw', 'bf', 'tf', 'Ap', 'Aw', 'Af', 'Z0', 'I', 'r', 'λ'],
        nargs='+',
        help='Data features'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='std',
        choices=['minmax', 'std'],
        help='choose normalization method.'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    predict = ULSPredict(args)
    history, prediction, initial_result, predict_result_pd, x_train, x_test, y_train, y_test = predict.model_compile()
    predict.inverse_norm_method(prediction)
    # predict.Excel_save(prediction)

if __name__ == '__main__':
    main()
