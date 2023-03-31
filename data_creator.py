import os
import string
import csv
from fontpreview import FontPreview
import pandas as pd
import requests, zipfile, io
import json
import shutil
#For a given font file, create the alphabet and the numbers 0-9
def create_alphabet(font_file, parent_folder):
    font = FontPreview(font_file)
    print(font.font.getname()[0])
    font_name = font.font.getname()[0]
    save_path = os.path.abspath(os.path.join(parent_folder, font_name))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #Loop through all the letters and create their images
    for char in string.ascii_letters:
        font.font_text = char
        font.bg_color = (255, 255, 255)  # white BG
        font.dimension = (512, 512)  # Dimension consistent with the default resolution for diffusion models
        font.fg_color = (0, 0, 0)  # Letter color
        font.set_font_size(300)  # font size ~ 300 pixels
        font.set_text_position('center')  # center placement

        if char in string.ascii_lowercase:
            image_file_name = 'lower_' + char + '.png'
        else:
            image_file_name = 'upper_' + char + '.png'
        font.save(os.path.abspath(os.path.join(save_path, image_file_name)))


    #Loop through all the digits and create their images
    for num in string.digits:
        font.font_text = num
        font.bg_color = (255, 255, 255)  # white BG
        font.dimension = (512, 512)  # Dimension consistent with the default resolution for diffusion models
        font.fg_color = (0, 0, 0)  # Letter color
        font.set_font_size(300)  # font size ~ 300 pixels
        font.set_text_position('center')  # center placement

        font.save(os.path.abspath(os.path.join(save_path, num + '.png')))

#Download the zip file, but only extract the requested file. Deletes the zip file when done!
def download_and_extract(df, save_folder):
    link = df['Link']
    requested_file_name = df['Filename']

    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extract(requested_file_name, path=os.path.abspath(save_folder))

#Uses pandas to read through the CSV from sheets without the need of constantly redownloading
def get_fonts(font_file_folder, font_image_folder):
    sheet_url = 'https://docs.google.com/spreadsheets/d/1VB10WsgaWIzCrovDXRroJTSkjH0z4ARo0PgrkAkyOI8/'
    sheet_url += 'export?format=csv&gid=0'
    df = pd.read_csv(sheet_url)


    head = df.head()

    for i in head.iterrows(): #Yeah yeah this is terribly ineffecient \
                                # but thats when its scaled to over 10k rows. we only have a couple 100
        download_and_extract(i[1], font_file_folder)


    fonts = os.listdir(os.path.abspath(font_file_folder))
    for f in fonts:
        file_path = os.path.abspath(os.path.join(font_file_folder, f))
        create_alphabet(file_path, font_image_folder)


#Create the jsonl file and training folder for the images
def create_dataset(font_image_path, font_file_path, training_data_path):

    # Step 1: Initialize the json file
    # Step 2: Loop through the Dataframe, for each row the Filename column corresponds to the actual
    #         folder name in 'font_images'.
    # Step 3: For each image in the respective folder, copy it over to the training folder (renaming it) and add its entry
    #         to the jsonl file

    #Step 1
    json_metadata = []
    if not os.path.exists(training_data_path):
        os.makedirs(training_data_path)


    #Step 2
    sheet_url = 'https://docs.google.com/spreadsheets/d/1VB10WsgaWIzCrovDXRroJTSkjH0z4ARo0PgrkAkyOI8/'
    sheet_url += 'export?format=csv&gid=0'
    df = pd.read_csv(sheet_url)

    head = df.head()

    file_name_counter = '0'
    for i in head.iterrows():
        row_data = i[1]

        folder_name = FontPreview(os.path.join(font_file_path, row_data['Filename'])).font.getname()[0]
        files_in_folder = os.listdir(os.path.abspath(os.path.join(font_image_path, folder_name)))
        for curr_file in files_in_folder:


            #JSON METADATA
            new_file_name = file_name_counter + '.png'
            font_characteristics = row_data['Descriptors']
            font_properties = row_data['Weight'] +' ' + row_data['Courner Rounding'] + ' ' + row_data['Serif']+ ' ' + row_data['Dynamics']  + ' ' + row_data['Width'] + ' ' + row_data['Capitals']

            #Determine prefix for metadata
            if 'lower' in str(curr_file):   #we know its an upper or lower letter
                prefix = 'A lowercase {} '.format(str(curr_file).replace('lower_', '').split('.')[0])


            if 'upper' in str(curr_file):
                prefix = 'An uppercase {} '.format(str(curr_file).replace('upper_', '').split('.')[0])


            if str(curr_file).split('.')[0].isdigit():
                prefix = 'The number {} '.format(str(curr_file).split('.')[0])


            font_text_data = prefix + 'which has traits ' + font_characteristics + ' and properties ' + font_properties


            #print(font_text_data)

            #Copy file to training location ~ Originally i was moving it
            #os.rename(os.path.abspath(os.path.join(font_image_path, folder_name, curr_file)), os.path.abspath(os.path.join(training_data_path, new_file_name)))
            shutil.copyfile(os.path.abspath(os.path.join(font_image_path, folder_name, curr_file)), os.path.abspath(os.path.join(training_data_path, new_file_name)))

            json_metadata.append( {'file_name': new_file_name, 'text':font_text_data }  )
            file_name_counter = str(int(file_name_counter) + 1)
        break


    #Create the jsonl file
    with open('metadata.jsonl', 'w') as f:
        for item in json_metadata:
            f.write(json.dumps(item) + '\n')




if __name__ == '__main__':
    #get_fonts('font_files', 'font_images')
    #create_dataset('font_images', 'font_files', 'train')
