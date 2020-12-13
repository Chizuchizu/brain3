#!/usr/bin/env python3

import glob
import os.path
import zipfile

zip_filename='submit.zip'
print(zip_filename)

def add_all(zip, files):
    for file in files:
        if os.path.isfile(file):
            zip.write(file, arcname=file)
            print('  ', file)

with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
    add_all(zip, glob.glob("src/**", recursive=True))
    add_all(zip, glob.glob("env.yaml"))
