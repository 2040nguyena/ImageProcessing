import pickle
import numpy as np
from PIL import Image
import os

# set the path to the pickled file
path_to_pickled_file1 = "./images/trains12G/train_data_batch_1"
path_to_pickled_file2 = "./images/trains12G/train_data_batch_2"
path_to_pickled_file3 = "./images/trains12G/train_data_batch_3"
path_to_pickled_file4 = "./images/trains12G/train_data_batch_4"
path_to_pickled_file5 = "./images/trains12G/train_data_batch_5"
path_to_pickled_file6 = "./images/trains12G/train_data_batch_6"
path_to_pickled_file7 = "./images/trains12G/train_data_batch_7"
path_to_pickled_file8 = "./images/trains12G/train_data_batch_8"
path_to_pickled_file9 = "./images/trains12G/train_data_batch_9"
path_to_pickled_file10 = "./images/trains12G/train_data_batch_10"

path_to_pickled_file_list = [path_to_pickled_file1,
                            path_to_pickled_file2,
                            path_to_pickled_file3,
                            path_to_pickled_file4,
                            path_to_pickled_file5,
                            path_to_pickled_file6,
                            path_to_pickled_file7,
                            path_to_pickled_file8,
                            path_to_pickled_file9,
                            path_to_pickled_file10,]


num = 0

# FIRST

# open the pickled file and load the data
with open(path_to_pickled_file1, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving first batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

# SECOND

# open the pickled file and load the data
with open(path_to_pickled_file2, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving second batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

#   THIRD

# open the pickled file and load the data
with open(path_to_pickled_file3, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving third batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

# FOURTH

# open the pickled file and load the data
with open(path_to_pickled_file4, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving FOURTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

#   FIFTH

# open the pickled file and load the data
with open(path_to_pickled_file5, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving FIFTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)









#   SIXTH

# open the pickled file and load the data
with open(path_to_pickled_file6, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving SIXTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

# SEVENTH

# open the pickled file and load the data
with open(path_to_pickled_file7, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving SEVENTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

#   EIGHTH

# open the pickled file and load the data
with open(path_to_pickled_file8, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving EIGHTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

# NINTH

# open the pickled file and load the data
with open(path_to_pickled_file9, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving NINTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)

#   TENTH

# open the pickled file and load the data
with open(path_to_pickled_file10, "rb") as f:
    data = pickle.load(f, encoding="bytes")

# reshape the image data from 1D to 3D arrays
images = data['data']
images = np.reshape(images, (-1, 3, 64, 64)).transpose(0, 2, 3, 1)
#labels = np.array(data[b"labels"])

# set the path to the output folder where the PNG files will be saved
output_folder_path = "./images_out/"

labels = data['labels']
# iterate over the images and save each one as a PNG file
print("saving TENTH batch")
for i in range(images.shape[0]):
    # get the label of the image
    label = labels[i]

    # save the image as a PNG file with the index as the filename
    image = Image.fromarray(images[i])
    output_path = os.path.join(output_folder_path, f"{num}.png")
    num += 1

    #   increase size of image before saving
    # Set the new size of the image
    new_size = (image.width * 8, image.height * 8)

    # Resize the image using the new size
    image = image.resize(new_size)

    image32 = Image.new('RGBA', image.size, (0, 0, 0, 0))
    image32.paste(image, (0, 0))

    image32.save(output_path)