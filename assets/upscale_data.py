from super_image import EdsrModel, ImageLoader
from PIL import Image

url = '/app/mission_ros_ws/src/visualnav-transformer-ros2/1.jpg'
image = Image.open(url)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=1.6).cuda()
inputs = ImageLoader.load_image(image).cuda()
preds = model(inputs)

ImageLoader.save_image(preds, './scaled_2x_1.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')