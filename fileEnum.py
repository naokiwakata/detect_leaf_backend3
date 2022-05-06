from enum import Enum


class File(Enum):
    Image = 1
    Video = 2

    def checkFileType(file):
        content_type = file.content_type
        if content_type == 'image/jpeg' or content_type == 'image/png':
            return File.Image
        else:
            return File.Video
