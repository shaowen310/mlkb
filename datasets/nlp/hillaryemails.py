import os
import itertools
import zipfile


class HillaryEmailsData:
    DATASET_NAME = 'hillary_emails'
    DATA_NAME = 'HillaryEmails.zip'

    def __init__(self, root='data_'):
        self.data_dir = os.path.join(root, self.DATASET_NAME)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.data_file = os.path.join(self.data_dir, self.DATA_NAME)

    @staticmethod
    def generate_samples(filepath):
        with zipfile.ZipFile(filepath, "r") as f:
            for name in f.namelist():
                _, filename = os.path.split(name)

                ext = os.path.splitext(filename)[-1].lower()
                if ext != '.txt':
                    continue

                data = f.read(name)

                yield filename, data


if __name__ == '__main__':
    d_emails = HillaryEmailsData()

    email_files = d_emails.generate_samples(d_emails.data_file)

    email_files = itertools.islice(email_files, 5)

    for name, data in email_files:
        print(name)
        print(data.splitlines()[:3])
