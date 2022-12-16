from PIL import Image
import os
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset


class Yoga_82(Dataset):
    """Clase que contiene la información del dataset Yoga-82.

    Args:
        root (str): ruta raiz donde se ubican las imágenes
        file_ (str): nombre del archivo a procesar
        transform (Compose): lista de transformaciones a aplicar a las imágenes
        prueba (bool): indica si es un dataset para hacer pruebas o no
        nivel (int): indica en nivel de clases a utilizar 6, 20 u 82
        n_samples (int): cantidad de datos a usar en el caso que sea un dataset de prueba
    """

    def __init__(self, root, file_, transform = True, prueba = True, nivel = 6, n_samples = None):
        # atributos de clase
        self.root = root
        self.file_ = file_
        self.nivel = nivel
        self.prueba = prueba
        self.transform = transform
        self.images = []
        self.labels_6 = []
        self.labels_20 = []
        self.labels_82 = []
        self.img_problemas = []

        if transform:
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomRotation((-10,10)),
                                                transforms.ToTensor()])


        # creación de diccionarios de distintas clases.
        with open('/content/Yoga-82/classes_6.txt', 'r') as f:
            self.classes_6 = {line.split(',')[0]: line.split(',')[1].replace('\n', '') for line in f}

        with open('/content/Yoga-82/classes_20.txt', 'r') as f:
            self.classes_20 = {line.split(',')[0]: line.split(',')[1].replace('\n', '') for line in f}

        self.classes_82 = {clase: i-1 for i, clase in enumerate(sorted(os.listdir('/content/Yoga-82/Images/'))) if clase != '.DS_Store'}

        
        # ruta del archivo
        path_file = os.path.join(root,file_)

        # guardar imágenes y etiquetas como atributos de clase
        with open(path_file, 'r') as f:
            for line in f:
                img, label_6, label_20, label_82 = line.split(',')
                try:
                    image_path = os.path.join(self.root, 'Images', img)
                    img_ = Image.open(image_path)
                    self.images.append(img)
                    self.labels_6.append(int(label_6))
                    self.labels_20.append(int(label_20))
                    self.labels_82.append(int(label_82))
                    # img.verify()
                except:
                    self.img_problemas.append(img)


        if self.prueba:
            idx = np.random.choice(np.arange(0, len(self.images)),n_samples)
            self.images = np.array(self.images)[idx]
            self.labels_6 = np.array(self.labels_6)[idx]
            self.labels_20 = np.array(self.labels_20)[idx]
            self.labels_82 = np.array(self.labels_82)[idx]

        
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'Images', self.images[idx])
        image = Image.open(image_path)
        if self.nivel == 6:
            label = self.labels_6[idx]
        elif self.nivel == 20:
            label = self.labels_20[idx]
        else:
            label = self.labels_82[idx]
        image = image.convert('RGB')

        # se realizan las transformaciones necesarias 
        # en caso que se necesiten
        if self.transform:
            image = self.transform(image)
        
        
        return image, label