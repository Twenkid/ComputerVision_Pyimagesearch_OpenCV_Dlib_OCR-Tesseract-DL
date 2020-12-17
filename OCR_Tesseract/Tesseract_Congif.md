https://medium.com/better-programming/beginners-guide-to-tesseract-ocr-using-python-10ecbb426c3d

"--psm 7" etc.

    0 : OSD_ONLY Orientation and script detection only.
    1 : AUTO_OSD Automatic page segmentation with orientation and script detection. (OSD)
    2 : AUTO_ONLY Automatic page segmentation, but no OSD, or OCR.
    3 : AUTO Fully automatic page segmentation, but no OSD. (default mode for tesserocr)
    4 : SINGLE_COLUMN-Assume a single column of text of variable sizes.
    5 : SINGLE_BLOCK_VERT_TEXT-Assume a single uniform block of vertically aligned text.
    6 : SINGLE_BLOCK-Assume a single uniform block of text.
    7 : SINGLE_LINE-Treat the image as a single text line.
    8 : SINGLE_WORD-Treat the image as a single word.
    9 : CIRCLE_WORD-Treat the image as a single word in a circle.
    10 : SINGLE_CHAR-Treat the image as a single character.
    11 : SPARSE_TEXT-Find as much text as possible in no particular order.
    12 : SPARSE_TEXT_OSD-Sparse text with orientation and script detection
    13 : RAW_LINE-Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
