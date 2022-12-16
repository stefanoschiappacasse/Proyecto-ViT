import torch


def predicciones_modelo(model, device, test_loader):
  model.eval()
  predicciones = []
  etiquetas = []
  
  for images, labels in test_loader:
    inputs = images.to(device)
    labels = labels.to(device)
      
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

    predicciones.extend(preds.cpu())
    etiquetas.extend(labels.cpu())  

  return etiquetas, predicciones