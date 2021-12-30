import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--size",
        default=0.2,
        type=float,
        choices=range(1),
        help="Test size percentage - [0-1)",
    )

    args = parser.parse_args()

    df_annotations = pd.read_pickle("../recommender/dataset/annotations_dataset.pickle")

    train, test = train_test_split(df_annotations, test_size=args.size)

    train.to_pickle("../recommender/dataset/annotations_dataset_train.pickle")
    test.to_pickle("../recommender/dataset/annotations_dataset_test.pickle")
