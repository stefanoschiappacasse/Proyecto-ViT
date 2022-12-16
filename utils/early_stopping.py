import numpy as np


class EarlyStopper:
  """
  Clase que controla el sobre ajuste observando
  cómo van variando las losses de validación 
  con las épocas. Se define la cantidad de épocas a esperar
  para parar luego de no haber mejoras (patience)
  y el margen a considerar (min_delta).
  """
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = np.inf

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
      return False
    elif validation_loss >= (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
      else:
        return False
    else:
      return False