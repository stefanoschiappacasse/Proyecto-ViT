import torch


def predicciones_modelo(model, device, loader):
  model.eval()
  predicciones = []
  etiquetas = []
  
  for images, labels in loader:
    inputs = images.to(device)
    labels = labels.to(device)
      
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

    predicciones.extend(preds.cpu())
    etiquetas.extend(labels.cpu())  

  return etiquetas, predicciones