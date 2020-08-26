import csv
import pandas as pd


class DataRetriever:
    def __init__(self):
        self.data = {'eng': []}

    @staticmethod
    def read_data(other_language, index, language):
        data = []
        file_path = f'data/fb_laser/tatoeba.{other_language}-eng.{language}'
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append((index, row[0], language))
                index += 1
        return data

    def load_data(self):
        for index, other_language in enumerate(['fra', 'spa', 'cmn']):
            self.data['eng'] += \
                self.read_data(other_language, 1000 * index, 'eng')
            self.data[other_language] = \
                self.read_data(other_language, 1000 * index, other_language)

    def store_data_as_csv(self, filename):
        columns = ['sentence_id', 'sentence', 'language']
        df = pd.DataFrame([], columns=columns)

        for lang_data in self.data.values():

            df = df.append(pd.DataFrame(lang_data, columns=columns),
                           ignore_index=True)
        df.to_csv(filename, index=False)


if __name__ == "__main__":
    data = DataRetriever()
    data.load_data()
    data.store_data_as_csv("data/raw_data.csv")

