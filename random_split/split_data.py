import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", default=0.2, type=int, help="Test size percentage")
    args = parser.parse_args()

    df_annotations = pd.read_json("../dataset/annotations_dataset.json", lines=True)

    train, test = train_test_split(df_annotations, test_size=args.size)

    train.to_json("../dataset/annotations_dataset_train.json", orient='records', lines=True)
    test.to_json("../dataset/annotations_dataset_test.json", orient='records', lines=True)
