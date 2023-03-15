import numpy as np

import cv2

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image

dog_classes = ['001.Affenpinscher',
 '002.Afghan_hound',
 '003.Airedale_terrier',
 '004.Akita',
 '005.Alaskan_malamute',
 '006.American_eskimo_dog',
 '007.American_foxhound',
 '008.American_staffordshire_terrier',
 '009.American_water_spaniel',
 '010.Anatolian_shepherd_dog',
 '011.Australian_cattle_dog',
 '012.Australian_shepherd',
 '013.Australian_terrier',
 '014.Basenji',
 '015.Basset_hound',
 '016.Beagle',
 '017.Bearded_collie',
 '018.Beauceron',
 '019.Bedlington_terrier',
 '020.Belgian_malinois',
 '021.Belgian_sheepdog',
 '022.Belgian_tervuren',
 '023.Bernese_mountain_dog',
 '024.Bichon_frise',
 '025.Black_and_tan_coonhound',
 '026.Black_russian_terrier',
 '027.Bloodhound',
 '028.Bluetick_coonhound',
 '029.Border_collie',
 '030.Border_terrier',
 '031.Borzoi',
 '032.Boston_terrier',
 '033.Bouvier_des_flandres',
 '034.Boxer',
 '035.Boykin_spaniel',
 '036.Briard',
 '037.Brittany',
 '038.Brussels_griffon',
 '039.Bull_terrier',
 '040.Bulldog',
 '041.Bullmastiff',
 '042.Cairn_terrier',
 '043.Canaan_dog',
 '044.Cane_corso',
 '045.Cardigan_welsh_corgi',
 '046.Cavalier_king_charles_spaniel',
 '047.Chesapeake_bay_retriever',
 '048.Chihuahua',
 '049.Chinese_crested',
 '050.Chinese_shar-pei',
 '051.Chow_chow',
 '052.Clumber_spaniel',
 '053.Cocker_spaniel',
 '054.Collie',
 '055.Curly-coated_retriever',
 '056.Dachshund',
 '057.Dalmatian',
 '058.Dandie_dinmont_terrier',
 '059.Doberman_pinscher',
 '060.Dogue_de_bordeaux',
 '061.English_cocker_spaniel',
 '062.English_setter',
 '063.English_springer_spaniel',
 '064.English_toy_spaniel',
 '065.Entlebucher_mountain_dog',
 '066.Field_spaniel',
 '067.Finnish_spitz',
 '068.Flat-coated_retriever',
 '069.French_bulldog',
 '070.German_pinscher',
 '071.German_shepherd_dog',
 '072.German_shorthaired_pointer',
 '073.German_wirehaired_pointer',
 '074.Giant_schnauzer',
 '075.Glen_of_imaal_terrier',
 '076.Golden_retriever',
 '077.Gordon_setter',
 '078.Great_dane',
 '079.Great_pyrenees',
 '080.Greater_swiss_mountain_dog',
 '081.Greyhound',
 '082.Havanese',
 '083.Ibizan_hound',
 '084.Icelandic_sheepdog',
 '085.Irish_red_and_white_setter',
 '086.Irish_setter',
 '087.Irish_terrier',
 '088.Irish_water_spaniel',
 '089.Irish_wolfhound',
 '090.Italian_greyhound',
 '091.Japanese_chin',
 '092.Keeshond',
 '093.Kerry_blue_terrier',
 '094.Komondor',
 '095.Kuvasz',
 '096.Labrador_retriever',
 '097.Lakeland_terrier',
 '098.Leonberger',
 '099.Lhasa_apso',
 '100.Lowchen',
 '101.Maltese',
 '102.Manchester_terrier',
 '103.Mastiff',
 '104.Miniature_schnauzer',
 '105.Neapolitan_mastiff',
 '106.Newfoundland',
 '107.Norfolk_terrier',
 '108.Norwegian_buhund',
 '109.Norwegian_elkhound',
 '110.Norwegian_lundehund',
 '111.Norwich_terrier',
 '112.Nova_scotia_duck_tolling_retriever',
 '113.Old_english_sheepdog',
 '114.Otterhound',
 '115.Papillon',
 '116.Parson_russell_terrier',
 '117.Pekingese',
 '118.Pembroke_welsh_corgi',
 '119.Petit_basset_griffon_vendeen',
 '120.Pharaoh_hound',
 '121.Plott',
 '122.Pointer',
 '123.Pomeranian',
 '124.Poodle',
 '125.Portuguese_water_dog',
 '126.Saint_bernard',
 '127.Silky_terrier',
 '128.Smooth_fox_terrier',
 '129.Tibetan_mastiff',
 '130.Welsh_springer_spaniel',
 '131.Wirehaired_pointing_griffon',
 '132.Xoloitzcuintli',
 '133.Yorkshire_terrier']

# transforms
data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def face_detector(pil_img):
    face_cascade = cv2.CascadeClassifier('static/trained_models/haarcascade_frontalface_alt.xml')

    # img = cv2.imread(pil_img)
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(pil_img):
    VGG16 = models.vgg16(pretrained=True)    
    
    img = data_transform(pil_img).float().unsqueeze(0)
    
    VGG16.cpu()
    VGG16.eval()
    output = VGG16(img)
    _, imgClassIndex = torch.max(output,1)

    if imgClassIndex >= 151 and imgClassIndex <= 268:
        return True
    else:
        return False 
    
def predict_breed_transfer(pil_img):
    model = models.resnet50(pretrained=True)

    n_inputs = model.fc.in_features
    last_layer = nn.Linear(n_inputs, 133)
    model.fc = last_layer

    model.load_state_dict(torch.load('static/trained_models/model_transfer.pt'))
    
    class_names = [item[4:].replace("_", " ") for item in dog_classes]
    # load the image and return the predicted breed
          
    img = data_transform(pil_img).float().unsqueeze(0)
    model.cpu()
    model.eval()
    output = model(img)
    _, preds = torch.max(output, 1)
    result = class_names[preds]
#     print('o',class_names[preds])
    
    return result

def run_app(pil_img):
    res = {}

    is_dog = dog_detector(pil_img)
    is_human = face_detector(pil_img)

    res['is_dog'] = is_dog
    res['is_human'] = is_human
    
    if is_dog and not is_human:
        dog_class = predict_breed_transfer(pil_img)
        res['dog_class'] = dog_class        
    elif is_human and not is_dog:
        dog_class = predict_breed_transfer(pil_img)
        res['dog_class'] = dog_class        
    else:
        res['dog_class'] = None

    return res