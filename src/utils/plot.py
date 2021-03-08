from PIL import Image, ImageDraw
from IPython.display import display
import random

def visualize_random_frames(frames_directory, bounding_boxes, frames_to_show=4):
	random_frames = random.choices(list(bounding_boxes.items()), k=frames_to_show)
	for key, boxes in random_frames:
		image = frames_directory / f'output{str(key+1).zfill(3)}.jpg'
		with Image.open(image) as im:
			for box in boxes:
				draw = ImageDraw.Draw(im)
				draw.rectangle([(box.xtl, box.ytl), (box.xbr, box.ybr)], outline='green', width=6)
			display(im)

#def create_iou_video(gt, preds, v)