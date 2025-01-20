from PIL import Image, ImageDraw, ImageFont
import os

def add_border_and_title(image, title, border_color=(255, 0, 0)):
    # Add a red border to the image
    bordered_image = Image.new('RGB', (image.width + 10, image.height + 10), border_color)
    bordered_image.paste(image, (5, 5))

    return bordered_image

# Get a list of all image files in the current directory
image_dir = r'C:\Users\ADMIN\Desktop\result\heng'
image_files = [file for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Separate images with 'rgb' and 'diff' endings
rgb_images = [image for image in image_files if 'rgb' in image]
diff_images = [image for image in image_files if 'diff' in image]

# Create a blank canvas for the final image
final_width = 0
final_height = 0
images = []

# Load and process RGB images
for rgb_image in rgb_images:
    image_path = os.path.join(image_dir, rgb_image)
    image = Image.open(image_path)
    target_height = 100  # Adjust as needed
    image = image.resize((int(image.width * target_height / image.height), target_height))
    image = add_border_and_title(image, os.path.basename(rgb_image))
    images.append(image)
    final_width += image.width + 30  # Add spacing between images

# Load and process diff images
for diff_image in diff_images:
    image_path = os.path.join(image_dir, diff_image)
    image = Image.open(image_path)
    target_height = 100  # Adjust as needed
    image = image.resize((int(image.width * target_height / image.height), target_height))
    image = add_border_and_title(image, os.path.basename(diff_image))
    images.append(image)
    final_width += image.width + 30  # Add spacing between images

# Calculate the final height of the canvas
num_rows = 2
final_height = (target_height + 50) * num_rows

# Create a blank canvas for the final image
final_image = Image.new('RGB', (final_width, final_height), color='white')
draw = ImageDraw.Draw(final_image)

# Paste images onto the canvas
current_x = 0
current_y = 0
for image in images:
    final_image.paste(image, (current_x, current_y))
    current_x += image.width + 30  # Add spacing between images
    if current_x + image.width + 30 > final_width:  # Move to the next row
        current_x = 0
        current_y += target_height + 50  # Add spacing between rows

# Draw borders around images
border_thickness = 5
draw.rectangle([5, 5, final_width - 5, final_height - 5], outline='black', width=border_thickness)

# Save or display the final image
final_image.save(os.path.join(image_dir, 'combined_image_with_borders_two_rows.jpg'))
final_image.show()
