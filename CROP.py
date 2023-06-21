from PIL import Image

tiff = Image.open('input_image.tif') #Open TIFF file

for i in range(tiff.n_frames): #Loop through all images in TIFF file
    
    tiff.seek(i) #Select current image 
    
    width, height = tiff.size  #Get size of current image
    
    left = (width - 50) // 2  #Define coordinates for the crop
    top = (height - 50) // 2
    right = left + 50
    bottom = top + 50
    
    cropped_image = tiff.crop((left, top, right, bottom)) #Crop image
    
    cropped_image.save(f'output_image_{i}.tif') #Save cropped image