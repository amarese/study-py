# -*- coding: utf-8 -*-
import os
import shutil
import uuid

base_path = "c:/basepath"

def main():
    for (path, dir, files) in os.walk(base_path):
        for filename in files:
            if filename.endswith(".jpg"):
                shutil.move(path + "/" + filename, path + "/" + uuid.uuid4().hex + ".jpg")

if __name__ == '__main__':
    main()
