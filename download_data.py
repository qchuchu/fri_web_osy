from zipfile import ZipFile
from os import rename, remove
from requests import get
from requests.exceptions import ConnectionError
from time import time, sleep
from tqdm import tqdm
import datetime

cs276_url = "http://web.stanford.edu/class/cs276/pa/pa1-data.zip"

print("Downloading the CS276 Corpus...")
start = time()
attempts = 0
while "cs276_request" not in globals():
    try:
        cs276_request = get(cs276_url, stream=True)
        total_size = int(cs276_request.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open("cs276.zip", "wb") as f:
            for data in cs276_request.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
    except ConnectionError as e:
        sleep(5 + attempts)
        attempts += 1
        print("Attempt {} failed".format(attempts))
end = time()
print("Corpus Downloaded!")
print(
    "Download duration : {}".format(str(datetime.timedelta(seconds=round(end - start))))
)
print("The corpus have been downloaded after : {} attempts".format(attempts))

with ZipFile("cs276.zip", "r") as zip_file:
    print("Extracting all the files now...")
    for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
        zip_file.extract(member=file, path="data/")
    print("Done!")

rename(r"data/pa1-data", r"data/cs276")
remove("cs276.zip")
