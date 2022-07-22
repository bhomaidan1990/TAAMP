from PIL import Image, ImageEnhance

#path_template = "{}/{}/{}"

#star = "star 1"
#task = "stir"
#file_names = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]

"""
file_names = ["task_5_zoom_in.png", "task_7_zoom_in.png"]

for file_name in file_names:
    #image_path = path_template.format(star, task, file_name)
    image_path = file_name
    colorImage  = Image.open(image_path)
    #enhancer = ImageEnhance.Brightness(colorImage)
    #factor = 1.7
    #factor = 1.2 # stir
    #processed_image = enhancer.enhance(factor)
    width, height = colorImage.size
    print("width = ", width)
    print("height = ", height)
    #left = int(width * 0.2)
    #right = width - left
    #top = int(height * 0.1)
    #bottom = height - top
    left = 300
    right = width - 200
    top = 250
    bottom = height - 300
    processed_image = colorImage.crop((left, top, right, bottom)) 
    processed_image.show()
    save_path = "processed/" + image_path
    processed_image.save(save_path, "PNG")

"""
file_name_template = "task_{}.png"
#file_names = [i for i in range(1, 9)]
file_names = [8]

for file_name in file_names:
    #image_path = path_template.format(star, task, file_name)
    image_path = file_name_template.format(file_name)
    colorImage  = Image.open(image_path)
    #enhancer = ImageEnhance.Brightness(colorImage)
    #factor = 1.7
    #factor = 1.2 # stir
    #processed_image = enhancer.enhance(factor)
    width, height = colorImage.size
    print("width = ", width)
    print("height = ", height)
    #left = int(width * 0.2)
    #right = width - left
    #top = int(height * 0.1)
    #bottom = height - top
    #left = 300
    #right = width - 250
    left = 350
    right = width - 300
    #top = 150
    top = 175
    #bottom = height - 150 # show the entire desk top
    bottom = height - 300
    processed_image = colorImage.crop((left, top, right, bottom)) 
    processed_image.show()
    save_path = "processed/" + image_path
    processed_image.save(save_path, "PNG")