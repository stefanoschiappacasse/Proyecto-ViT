import torch


def predicciones_modelo(model, device, loader):
  """Función que genera las predicciones de un modelo sobre un 
  conjunto de datos

  Args:
      model (pytorch model): modelo a utilizar para generar las predicciones.
      device (str): dispositivo donde realizar la computación de las predicciones.
      loader (data loader): objeto dataloader de pytorch que contiene los datos a usar.

  Returns:
      predicciones y etiquetas
  """
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