# import wandb
import time
import copy
import torch
import argparse
import logging
from torch import nn

from ViT.ViT_model import ViT
from ViT.ViT_parts import *
from utils.preprocesamiento import redimensionamiento_imagenes
from utils.data_loading import create_dataset, create_dataloaders
from utils.early_stopping import EarlyStopper




def train_model(model, 
                criterion, 
                optimizer,
                scheduler, 
                chkp_path,
                device,
                train_loader,
                val_loader,
                num_epochs = 25):
  """
  Función que aplica el entrenamiento del modelo, 
  dado un criterio de loss, un optimizador y la cantidad de épocas.
  Retorna el mejor modelo encontrado, en conjunto con las losses de 
  entrenamiento y validación, como también los accuracy para ambos conjuntos.
  """
  since = time.time()
  
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_acc_train = 0.0
  val_acc = []
  train_acc = []
  val_loss = []
  train_loss = []
  
  early_stopper = EarlyStopper(patience=5)

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)
    
    #Train model
    scheduler.step()
    model.train()
    
    running_loss = 0.0
    running_corrects = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
          
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
      
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    epoch_acc_train = epoch_acc
    

    #Validation 
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
          
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
     
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

      
    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    val_loss.append(epoch_loss)
    val_acc.append(epoch_acc)
    
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_acc_train = epoch_acc_train
        best_model_wts = copy.deepcopy(model.state_dict())

    if early_stopper.early_stop(epoch_loss): 
        break
    if (epoch + 1)%5 == 0:
        torch.save(model.state_dict(), chkp_path + '_' + str((epoch + 1)) +'.pth')
        
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, 
                                                      time_elapsed % 60))
  print('Best train accucary: {:.4f}'.format(best_acc_train))
  print('Best val accucary: {:.4f}'.format(best_acc))

  model.load_state_dict(best_model_wts)
  return model, train_loss, train_acc, val_loss, val_acc


def get_args():
    parser = argparse.ArgumentParser(description='Entrenamiento ViT con dataset Yoga-82')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', metavar='B', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_heads', '-nh', metavar='NH', type=int, default=12, help='Number of heads')
    parser.add_argument('--emb_size', '-nh', metavar='ES', type=int, default=768, help='Embedding size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.2,
                        help='Porcentaje de datos a usar para el conjunto de validación')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes', dest = 'num_classes')
    parser.add_argument('--rooth_path', '-rp', type=str, default='/PROYECTO-VIT/data/', help='Root path', dest = 'root_path')
    parser.add_argument('--file_path', '-fp', type=str, default='/PROYECTO-VIT/data/yoga_train.txt', help='File path', 
                        dest = 'file_path')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    root_path = args.rooth_path
    file_path = args.file_path
    model = ViT(n_classes=args.classes, emb_size = args.emb_size, num_heads = args.num_heads)
    


    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.load.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    redimensionamiento_imagenes(root_path)

    # crear dataset      
    train_dataset = create_dataset(root_path, file_path, True, True, 6)
    # crear dataloaders de train y de test

    # crear data loaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset = None, set_ = 'train', 
                                                 batch_size = args.batch_size, val_size =args.val)

    try:
        mynet, mynet_tl, mynet_ta, mynet_vl, mynet_va = \
        train_model(model = model, 
                    criterion = criterion, 
                    optimizer = optimizer, 
                    scheduler = scheduler,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    num_epochs = 20)

    except:
        pass