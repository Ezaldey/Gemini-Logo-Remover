'''Note: I made this icon script with chatgpt '''
from PIL import Image, ImageDraw, ImageFilter
import math

SIZE = 512

img = Image.new("RGBA", (SIZE, SIZE), (20, 20, 28, 255))
draw = ImageDraw.Draw(img)

# Rounded dark app background
draw.rounded_rectangle(
    (20, 20, SIZE-20, SIZE-20),
    radius=110,
    fill=(30, 30, 43, 255)
)

# Subtle glow circle
glow = Image.new("RGBA", img.size, (0,0,0,0))
g = ImageDraw.Draw(glow)
g.ellipse((90, 80, 420, 410), fill=(91,141,239,70))
glow = glow.filter(ImageFilter.GaussianBlur(45))
img.alpha_composite(glow)

draw = ImageDraw.Draw(img)

# Image frame
draw.rounded_rectangle(
    (105, 135, 407, 370),
    radius=35,
    fill=(242,242,247,255)
)

# Mountain/photo inside
draw.polygon(
    [(130,335),(210,250),(265,310),(320,235),(385,335)],
    fill=(91,141,239,255)
)

draw.ellipse(
    (310,175,355,220),
    fill=(165,102,255,255)
)

# Removed logo area (transparent-looking checker)
for y in range(250,310,25):
    for x in range(145,205,25):
        color = (210,210,220,255) if (x//25+y//25)%2 else (230,230,240,255)
        draw.rectangle((x,y,x+25,y+25), fill=color)

# Eraser symbol
draw.rounded_rectangle(
    (290,300,390,365),
    radius=20,
    fill=(61,220,151,255)
)

draw.polygon(
    [(300,300),(350,250),(410,310),(360,360)],
    fill=(61,220,151,255)
)

# AI sparkle
def sparkle(cx, cy, r):
    pts=[]
    for i in range(8):
        angle=i*math.pi/4
        radius=r if i%2==0 else r/4
        pts.append(
            (
                cx+math.cos(angle)*radius,
                cy+math.sin(angle)*radius
            )
        )
    draw.polygon(pts, fill=(255,180,84,255))

sparkle(390,120,45)
sparkle(335,95,20)

# Save ICO with multiple resolutions
img.save(
    "AI_Logo_Remover.ico",
    sizes=[
        (16,16),
        (32,32),
        (48,48),
        (64,64),
        (128,128),
        (256,256),
        (512,512)
    ]
)

print("Created AI_Logo_Remover.ico")
