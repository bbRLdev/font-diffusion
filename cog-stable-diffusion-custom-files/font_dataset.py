import os
import json
import re

from torch.utils.data import Dataset

from PIL import Image


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class font_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'sd-test-metadata.jsonl','test':'sd-test-metadata.jsonl'}
        
        # data = [json.loads(line) for line in open('data.json', 'r')]
        # self.annotation = json.loads(open(os.path.join(ann_root,filenames[split]),'r'))
        self.annotation = [json.loads(line) for line in open(os.path.join(ann_root,filenames[split]),'r')]
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['file_name'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['text']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['file_name'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
                                    


class font_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename = 'sd-train-metadata.jsonl'

        
        # self.annotation = json.loads(open(os.path.join(ann_root,filename),'r'))
        self.annotation = [json.loads(line) for line in open(os.path.join(ann_root,filename),'r')]

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['file_name'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
#         print(self.max_words)
#         print(len(ann['text']))
        
        caption = self.prompt+pre_caption(ann['text'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
