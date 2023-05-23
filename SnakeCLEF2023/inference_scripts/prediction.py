import sys
sys.path.append('/absolute-path/snakeCLEF/training_scripts')  #path where src folder is located.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from src.core import models, training, data
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', type=str, default='../../', help='path to test data directory')
parser.add_argument('--model_arch', type=str, default='deit_base_distilled_384', help='name of model architecture')
parser.add_argument('--model_name', type=str, default='clef2023_deit_base_distilled_384_ensemble_focal_05-15-2023_12-27-11', help='name of model file')
parser.add_argument('--data_csv', type=str, default='../../snake_csv_files/SnakeCLEF2023-PubTestMetadata.csv', help='path to test csv file')
parser.add_argument('--model_path', type=str, default='../results/models/', help='path to models folder')

args = parser.parse_args()


DATA_DIR = args.data_dir
MODEL_ARCH = args.model_arch
MODEL_NAME = args.model_name


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# load metadata
test_df = pd.read_csv(args.data_csv)

print(f'Test set length: {len(test_df):,d}')

#test dataset 

class SnakeInferenceDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = self.data.iloc[index]
        img = Image.open(DATA_DIR+image.image_path).convert("RGB")

        if transform is not None:
            img = self.transform(img)

        return img
      
      
# create fine-tuned network
model = models.get_model(MODEL_ARCH, 1784, pretrained=False)
training.load_model(model, MODEL_NAME, path=args.model_path)
assert np.all([param.requires_grad for param in model.parameters()])

model_config = model.pretrained_config
batch_size = 128

# create transforms
_, test_tfms = data.get_transforms(
    size=model_config['input_size'], mean=model_config['image_mean'],
    std=model_config['image_std'])

from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])


prediction_list = []

test_dataset = SnakeInferenceDataset(test_df, transform = transform) 
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

loop=tqdm(test_dataloader)
for batch, X in enumerate(loop):

  device = torch.device(device)
  X = X.to(device)
  
  with torch.no_grad():
    preds = model(X)
    final_result = torch.argmax(preds, axis=1)
    prediction_list.append(final_result.tolist())
    
 

flat_list = [item for sublist in prediction_list for item in sublist]
df_prediction = pd.DataFrame(flat_list)
df_prediction.to_csv('test_prediction.csv')
df_prediction.columns = ['class_id']

final_df = pd.concat([ test_df['observation_id'], df_prediction], axis=1)
df = final_df.drop_duplicates('observation_id', keep='last')
df.to_csv('snake_prediction.csv', index=False)

