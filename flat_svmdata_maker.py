import sys
from multiprocessing import Process
import os.path
import pandas as pd
import audio_processor as ap

annotations="./data/annotations_final.csv"

train_svmdata = "mtg.train.svmdata"


def main(num_unit, unit_number):
    df = pd.read_csv(annotations, delimiter="\t")

    with open(train_svmdata + "%d_%d" %(unit_number, num_unit), "w") as io:
        for row, path in enumerate(df['mp3_path']):
            if row % num_unit != unit_number:
                continue

            path = os.path.join("./data/mp3", path)
            print(unit_number, row, path)
            melgram = ap.compute_melgram(path)
            rows = melgram.shape[2]
            cols = melgram.shape[3]
            for i in range(rows):
                for j in range(cols):
                    index = i * cols + j + 1
                    v = melgram[0,0,i,j]
                    io.write("%d:%f " %(index, v))
            io.write("\n")


if __name__ == '__main__':

    num_unit = int(sys.argv[1])
    unit_number = int(sys.argv[2])

    main(num_unit, unit_number)
