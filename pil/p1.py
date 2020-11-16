#Example: Draw Partial Opacity Text
from PIL import Image, ImageDraw, ImageFont, ImageGrab

s = ImageGrab.grab(bbox=None)

s.show()

# get an image
base = Image.open(r"C:\Users\user\Pictures\Saved Pictures\123.png").convert('RGBA')

# make a blank image for the text, initialized to transparent text color
txt = Image.new('RGBA', base.size, (255,255,255,0))

# get a font
fnt = ImageFont.truetype("C:\Program Files\Python36\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\DejaVuSerif.ttf",40);

 #'Pillow/Tests/fonts/FreeMono.ttf', 40)
# get a drawing context
d = ImageDraw.Draw(txt)

# draw text, half opacity

x = 200; y = 200;
d.text((x,y), "Hello", font=fnt, fill=(255,255,255,128))
# draw text, full opacity
d.text((10,60), "World", font=fnt, fill=(255,255,255,255))

out = Image.alpha_composite(base, txt)

out.show()

out.save("pilpil.png")

