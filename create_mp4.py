from io import BytesIO
from moviepy.editor import ImageSequenceClip
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def load_frames_from_directory(directory, input = False):
    frames = []
    frames_paths = os.listdir(directory)
    # print(sorted(frames_paths))

    frames_paths.sort(key=natural_keys)
    print(frames_paths)
    for f in frames_paths:
        frame_path = os.path.join(directory, f)  # Assuming the frames are named as 'frame0.png', 'frame1.png', etc.
        frame = Image.open(frame_path)
        if input:
            frame = frame.resize((640, 320), Image.ANTIALIAS)
        frames.append(frame)
    return frames

frames1 = load_frames_from_directory('/home/sandra/Documents/diod-main/demo_cvpr_supp_mat/Input', 0)
frames2 = load_frames_from_directory('/home/sandra/Documents/diod-main/demo_cvpr_supp_mat/DOM')
frames3 = load_frames_from_directory('/home/sandra/Documents/diod-main/demo_cvpr_supp_mat/BMOD')
frames4 = load_frames_from_directory('/home/sandra/Documents/diod-main/demo_cvpr_supp_mat/DIOD')


# Define caption texts for each frame set
captions = ["Input sequence", "Bao et al. [1]", "BMOD [15]", "DIOD (ours)"]

# Define the size of the caption area
caption_height = 30  

# Define a font for the caption text with the new size
try:
    # Use a nice font
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=30)
except IOError:
    # Fallback to the default PIL font if it's not found
    font = ImageFont.load_default()

# Define margins
horizontal_margin = 10  # Space between images horizontally
vertical_margin = 10    # Space between images vertically

# Determine the size of the combined image
frame_width = max(frame.width for frame in frames1)  
frame_height = max(frame.height for frame in frames1)  

# The width of the canvas should be the width of three frames plus two horizontal margins
canvas_width = 3 * frame_width + 2 * horizontal_margin

# The height of the canvas should be twice the frame height plus a vertical margin and caption heights
canvas_height = 2 * frame_height + vertical_margin + 2 * caption_height

# Create a list to hold the combined frames
combined_clips = []
frame_arrays = []

for i in range(len(frames1)):
    # Create a new blank image with a white background
    new_im = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    # Paste the single frame on the top row, centered
    new_im.paste(frames1[i], (canvas_width // 2 - frame_width // 2, 0))

    # Paste the three frames on the second row
    new_im.paste(frames2[i], (0, frame_height + vertical_margin))
    new_im.paste(frames3[i], (frame_width + horizontal_margin, frame_height + vertical_margin))
    new_im.paste(frames4[i], (2 * frame_width + 2 * horizontal_margin, frame_height + vertical_margin))

    # Initialize ImageDraw to add text
    draw = ImageDraw.Draw(new_im)

    y_offset = 100

    # Draw the caption for the single frame on the first row, centered
    draw.text(
        (canvas_width // 2 - draw.textsize(captions[0], font=font)[0] // 2, frame_height - y_offset + 30),
        captions[0],
        font=font,
        fill="black"
    )

    # Draw the captions for the three frames on the second row
    draw.text((frame_width // 2 - draw.textsize(captions[1], font=font)[0] // 2, canvas_height - caption_height- y_offset), captions[1], font=font, fill="black")
    draw.text((3 * frame_width // 2 + horizontal_margin - draw.textsize(captions[2], font=font)[0] // 2, canvas_height - caption_height - y_offset), captions[2], font=font, fill="black")
    draw.text((5 * frame_width // 2 + 2 * horizontal_margin - draw.textsize(captions[3], font=font)[0] // 2, canvas_height - caption_height - y_offset), captions[3], font=font, fill="black")

    # Save the combined image to a file-like object
    combined_image_io = BytesIO()
    new_im.save(combined_image_io, format='PNG')
    combined_image_io.seek(0)

    # Convert BytesIO back to a PIL image
    pil_image = Image.open(combined_image_io)

    # Convert PIL image to numpy array and append to the list
    frame_arrays.append(np.array(pil_image))

# Use moviepy to create a clip from this combined image
clip = ImageSequenceClip(frame_arrays, fps=5)  # fps=1 if we want each image to show for 1 second

# Write the result to a file
output_path = '/home/sandra/Documents/diod-main/demo_cvpr_supp_mat/combined_animation3.mp4'
clip.write_videofile(output_path, codec='libx264')