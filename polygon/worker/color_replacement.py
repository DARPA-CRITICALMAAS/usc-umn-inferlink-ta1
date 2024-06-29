import cv2
import numpy as np
import os


# Load the image in LAB color space
def load_image(image_path):
    image = cv2.imread(image_path)

    image_size = image.shape
    image = cv2.resize(image, (image_size[1]//2, image_size[0]//2), interpolation=cv2.INTER_NEAREST)
    #if image_size[0] > 1500:
    #    image = image[1100:1500, 1200:1800, :]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv_image_print = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    #cv2.imwrite('Example_Processing/CO_DenverW_processed_v001_source.tif', hsv_image_print)
    return hsv_image, np.unique(hsv_image[:, :, 0]).shape[0], image_size

# Pad the image with NaN values
def pad_image(image, pad_size):
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    padded_image = np.pad(image.data, pad_width, mode='constant', constant_values=np.nan)
    padded_mask = np.pad(image.mask, pad_width, mode='constant', constant_values=True)
    return np.ma.masked_array(padded_image, mask=padded_mask)

# Threshold the image to identify black colors
def threshold_black(image, threshold=90, dilation_size=3):
    V_channel = image[:,:,2]
    mask_black = V_channel < threshold

    #print(mask_black.shape)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black.astype(np.uint8), kernel, iterations=1)
    #mask_black_dilated_print = mask_black_dilated
    #mask_black_dilated_print[mask_black_dilated_print>0] = 255
    #cv2.imwrite('Example_Processing/CO_DenverW_processed_v001_black_'+str(threshold)+'.tif', mask_black_dilated)
    return mask_black_dilated


def generate_shifted_images(image, pad_size):
    h, w, c = image.shape
    shifts = np.arange(-pad_size//2, pad_size//2 + 1)
    shifted_images = np.zeros((h - 2*pad_size, w - 2*pad_size, c, len(shifts)**2), dtype=image.dtype)
    mask_images = np.zeros((h - 2*pad_size, w - 2*pad_size, c, len(shifts)**2), dtype=bool)
    
    index = 0
    for dx in shifts:
        for dy in shifts:
            shifted_image = np.roll(np.roll(image.data, dy, axis=0), dx, axis=1)
            shifted_mask = np.roll(np.roll(image.mask, dy, axis=0), dx, axis=1)
            shifted_images[:, :, :, index] = shifted_image[pad_size:-pad_size, pad_size:-pad_size]
            mask_images[:, :, :, index] = shifted_mask[pad_size:-pad_size, pad_size:-pad_size]
            index += 1
    return np.ma.masked_array(shifted_images, mask=mask_images)


# Select the most common color excluding NaN
def most_common_color(channels):
    #print(channels)
    reshaped = channels.reshape(-1, channels.shape[-1])
    masked = np.ma.masked_invalid(reshaped)
    
    def bincount_with_mask(arr):
        arr = arr.compressed().astype(int)
        if arr.size == 0:
            return np.array([0])  # Return 0 if all values are NaN
        return np.bincount(arr, minlength=256)
    
    counts = np.apply_along_axis(bincount_with_mask, axis=1, arr=masked)
    #print(counts)
    most_common = np.argmax(counts, axis=1)
    #print(most_common)
    #most_common[masked.mask.all(axis=1)] = 0  # Assign black color if all channels are NaN
    return most_common.reshape(channels.shape[:2])




# Replace black pixels with most common nearby color
def replace_black_pixels(image, pad_size=10, threshold=90, dilation_size=3):
    #print(image.shape)
    mask_black = threshold_black(image, threshold, dilation_size)
    masked_image = np.ma.masked_array(image, mask=np.repeat(mask_black[:, :, np.newaxis], 3, axis=2))
    #print(masked_image.shape)
    
    padded_image = pad_image(masked_image, pad_size)
    shifted_images = generate_shifted_images(padded_image, pad_size)
    #print(shifted_images.shape)
    
    new_image = np.zeros_like(image)
    for channel in range(3):
        #print('processing channel... ', channel)
        shifted_channel_images = shifted_images[:, :, channel, :]
        most_common_channel = most_common_color(shifted_channel_images)
        new_image[:,:,channel] = np.where(mask_black, most_common_channel, image[:,:,channel])
    
    return new_image


# Save the output image
def save_image(image, output_path, image_size):
    image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, bgr_image)


def color_replacement(input_image_dir, output_image_dir, this_image_name):

    input_image_path = os.path.join(input_image_dir, this_image_name)
    output_image_path = os.path.join(output_image_dir, this_image_name.replace('.png', '_recolored_v1.png'))
    
    lab_image, unique_counting, image_size = load_image(input_image_path)
    if unique_counting <= 2:
        recolored_image = lab_image
    else:
        try:
            recolored_image = replace_black_pixels(lab_image, pad_size=10)
        except:
            recolored_image = lab_image
    save_image(recolored_image, output_image_path, image_size)

    input_image_path = os.path.join(output_image_dir, this_image_name.replace('.png', '_recolored_v1.png'))

    
    output_image_path = os.path.join(output_image_dir, this_image_name.replace('.png', '_recolored_v2.png'))

    lab_image, unique_counting, image_Size = load_image(input_image_path)
    if unique_counting <= 2:
        recolored_image = lab_image
    else:
        try:
            recolored_image = replace_black_pixels(lab_image, pad_size=10, threshold=150, dilation_size=3)
        except:
            recolored_image = lab_image
    save_image(recolored_image, output_image_path, image_size)
    

    return this_image_name, True