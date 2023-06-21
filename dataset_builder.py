import openpyxl
import os
from PIL import Image

# wafer name to load
wafer_name = "DP57572.1Y_09"

# open the excel file name "Labeled_defect_2.xlsx"
wb = openpyxl.load_workbook("Labeled_defect_2.xlsx")

# funtion that takes an index number, returns a cropped image from a TIFF file
def crop_image(imgIndex, wname):
    tiff = Image.open(wname+".tif")

    tiff.seek(imgIndex-1)
   
    # Define the coordinates for the crop
    xloc = tiff.size[0]/2
    yloc = tiff.size[1]/2
    
    # Crop the image
    croppedImage = tiff.crop((xloc-35, yloc-35, xloc+35, yloc+35))

    return croppedImage

# read sheet name "Defect Label" and extract all value of 1st row  
sheet = wb["Defect Label"]
row = sheet[1]
 
for idx, cell in enumerate(row):
    if cell.value == wafer_name:
        print("Exist")
        col_val =openpyxl.utils.cell.get_column_letter(idx+1)
        print(col_val)

# for column index "col_val" print all row values to end

for row in sheet[col_val][3:]:
    if row.value == "Wafer Back side" or row.value == "Wafer back side":
        break

    elif row.value == "Contamination/Particle":
        row.value = "Contamination-Particle"
        img_label = 'A'
    elif row.value == "Pattern defect":
        img_label = 'B'
    elif row.value == "Probe Mark":
        img_label = 'C'
    elif row.value == "Scratches":
        img_label = 'D'
    elif row.value == "Others":
        img_label = 'E'
    else:
        pass

    #create a folder using name contains in row.value and create one if it doesn't exist
    path = os.path.join(os.getcwd(), str(row.value))

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    row_index = row.row-3
    
    # crop image and save it to the created folder
    img = crop_image(row_index, wafer_name)

    filename = wafer_name+"_"+str(row_index)+"_"+img_label+".png"
    img.save(path+"/"+filename)

    print(str(row.value)+ " "+str(row_index))

#print(path) 
#img = crop_image(7,wafer_name)
#img.save(path+"/"+img_label+".png")