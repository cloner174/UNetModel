import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datasets import FullyAnnotatedDataset, WeaklyAnnotatedDataset
from UNet import MultitaskAttentionUNet
from utils import visualize_predictions, pro_prediction
from losses import CombinedLoss
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from collections import OrderedDict
from multiprocessing import Pool, cpu_count, set_start_method
from typing import Callable, List, Tuple

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
if device.__repr__().split("'")[-2] == 'cpu':
    pass
else:
    capability = torch.cuda.get_device_capability(device)
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f"{capability[0]}.{capability[1]}")
  

#
base_dir = '/content/Drive/MyDrive/LunaProject' #/main

data = base_dir + '/Final_Images_and_Masks.zip'

X = np.load(base_dir + '/X.npy')
masks = np.load(base_dir + '/masks.npy')
boxes = np.load(base_dir + '/y_centroids.npy')
y_class = np.load(base_dir + '/y_class.npy')

X_weak = np.load(base_dir + '/images.npy')
y_train_weak = np.load(base_dir + '/labels.npy')


X_train, X_val, masks_train, masks_val, boxes_train, boxes_val, y_train, y_val = train_test_split(
        X , masks, boxes, y_class,
        test_size=0.2, random_state=42, shuffle=True
)

annotated_dataset = FullyAnnotatedDataset(X_train, masks_train, boxes_train, y_train, transform=None)

weak_dataset = WeaklyAnnotatedDataset(X_weak, y_train_weak, transform=None)

val_dataset = FullyAnnotatedDataset(X_val, masks_val, boxes_val, y_val, transform=None)

annotated_loader = DataLoader(annotated_dataset, batch_size=64, shuffle=True, num_workers=8)

weak_loader = DataLoader(weak_dataset, batch_size=64, shuffle=True, num_workers=8)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)


model = MultitaskAttentionUNet(input_channels=1, num_classes=1, bbox_size=4)

model.load_state_dict(torch.load(os.path.join('./', 'UNet.pth'), weights_only=True))

criterion_seg = CombinedLoss()
criterion_cls = nn.BCEWithLogitsLoss()
criterion_loc = nn.MSELoss()

#
optimizer = optim.Adam(model.parameters(), lr=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.4, patience=4)
#
base_dir = './'
num_epochs = 5
best_val_loss = float('inf')


c1 = 1.5
c2 = 1.5
w = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

writer = SummaryWriter('runs/pso_experiment')

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute the Dice Coefficient between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted masks.
        target (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Mean Dice score across the batch.
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()



def fitness_function(model: torch.nn.Module, data_loader: DataLoader) -> float:
    """
    Evaluate the model's performance using the Dice coefficient.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader providing the evaluation data.
    
    Returns:
        float: Average Dice score over the dataset.
    """
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for batch_idx, (images, masks, _, _) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs, *rest = model(images)
            outputs = torch.sigmoid(outputs)
            
            preds = (outputs > 0.5).float()
            dice = dice_coefficient(preds, masks)
            dice_scores.append(dice)
    
    average_dice = np.mean(dice_scores)
    return average_dice



def fitness_function_wrapper(args: Tuple[torch.nn.Module, DataLoader]) -> float:
    model, data_loader = args
    return fitness_function(model, data_loader)


class Particle:
    """
    Represents a single particle in the PSO swarm.
    
    Attributes:
        model (torch.nn.Module): The current model state.
        velocity (OrderedDict): Velocities for each parameter.
        best_model (torch.nn.Module): Personal best model state.
        best_score (float): Best fitness score achieved by this particle.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model)
        self.model.to(device)
        self.velocity = OrderedDict(
            (name, torch.zeros_like(param.data, device=device)) for name, param in self.model.named_parameters()
        )
        self.best_model = copy.deepcopy(model).to(device)
        self.best_score = -np.inf
    
    def update_velocity(
        self,
        global_best_state: OrderedDict,
        w: float,
        c1: float,
        c2: float
    ):
        """
        Update the particle's velocity based on inertia, cognitive, and social components.
        
        Args:
            global_best_state (OrderedDict): Global best model state.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
        """
        for name, param in self.model.named_parameters():
            r1 = torch.rand_like(param.data, device=device)
            r2 = torch.rand_like(param.data, device=device)
            
            personal_best_param = self.best_model.state_dict()[name]
            global_best_param = global_best_state[name]
            
            cognitive = c1 * r1 * (personal_best_param - param.data)
            social = c2 * r2 * (global_best_param - param.data)
            
            self.velocity[name] = w * self.velocity[name] + cognitive + social
            
            vel_std = self.velocity[name].std().item()
            clamp_val = max(0.1, vel_std * 2)  # Ensure clamp_val is at least 0.1
            self.velocity[name].clamp_(-clamp_val, clamp_val)
    
    def update_position(self):
        """
        Update the particle's position by adjusting model parameters with the current velocity.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data += self.velocity[name]
                
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                clamp_min = param_mean - 3 * param_std
                clamp_max = param_mean + 3 * param_std
                param.data.clamp_(clamp_min, clamp_max)
    
    def evaluate_fitness(self, fitness_func: Callable, data_loader: DataLoader) -> float:
        """
        Evaluate the particle's fitness and update personal best if necessary.
        
        Args:
            fitness_func (Callable): Function to compute fitness.
            data_loader (DataLoader): DataLoader providing the evaluation data.
        
        Returns:
            float: Current fitness score.
        """
        current_score = fitness_func(self.model, data_loader)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_model = copy.deepcopy(self.model)
        
        return current_score



class PSO:
    """
    Implements Particle Swarm Optimization for model parameter tuning.
    
    Attributes:
        swarm (List[Particle]): List of particles in the swarm.
        global_best_model (torch.nn.Module): Global best model across the swarm.
        global_best_score (float): Best fitness score achieved globally.
        fitness_func (Callable): Function to compute fitness.
        data_loader (DataLoader): DataLoader for fitness evaluation.
        max_iterations (int): Maximum number of iterations.
        w (float): Initial inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        verbosity (int): Verbosity level for logging.
        early_stopping_rounds (int): Number of iterations with no improvement to trigger early stopping.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        num_particles: int,
        fitness_func: Callable,
        data_loader: DataLoader,
        max_iterations: int = 50,
        w: float = 0.9,
        c1: float = 2.0,
        c2: float = 2.0,
        verbosity: int = 1,
        early_stopping_rounds: int = 10
    ):
        self.swarm: List[Particle] = [Particle(model) for _ in range(num_particles)]
        self.global_best_model = copy.deepcopy(self.swarm[0].model).to(device)
        self.global_best_score = -np.inf
        self.fitness_func = fitness_func
        self.data_loader = data_loader
        self.max_iterations = max_iterations
        self.initial_w = w
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbosity = verbosity
        self.early_stopping_rounds = early_stopping_rounds
        self.no_improvement_count = 0
        self.fitness_history = []
    
    def evaluate_swarm_fitness(self) -> List[float]:
        """
        Evaluate the fitness of all particles in the swarm in parallel.
        
        Returns:
            List[float]: List of fitness scores for each particle.
        """
        with Pool(processes=cpu_count()) as pool:
            fitness_args = [(particle.model, self.data_loader) for particle in self.swarm]
            fitness_scores = pool.map(fitness_function_wrapper, fitness_args)
        return fitness_scores
    
    def step(self):
        """
        Execute the PSO optimization process with enhanced features.
        """
        print('Starting PSO optimization...')
        for iteration in range(1, self.max_iterations + 1):
            print(f'PSO Iteration [{iteration}/{self.max_iterations}]')
            
            fitness_scores = self.evaluate_swarm_fitness()
            
            for idx, (particle, score) in enumerate(zip(self.swarm, fitness_scores)):
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_model = copy.deepcopy(particle.model)
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_model = copy.deepcopy(particle.model)
                
                if self.verbosity > 1:
                    logger.debug(f'Particle {idx + 1}: Current Score = {score:.4f}, '
                                 f'Personal Best = {particle.best_score:.4f}')
            
            self.fitness_history.append(self.global_best_score)
            writer.add_scalar('Global Best Score', self.global_best_score, iteration)
            print(f'Current Global Best Score: {self.global_best_score:.4f}')
            
            if iteration > 1 and self.fitness_history[-1] <= self.fitness_history[-2]:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.early_stopping_rounds:
                    print(f'No improvement in the last {self.early_stopping_rounds} iterations. '
                                f'Early stopping triggered.')
                    break
            else:
                self.no_improvement_count = 0
            
            decay_rate = 0.95 
            self.w *= decay_rate
            self.w = max(self.w, 0.4) 
            logger.debug(f'Updated inertia weight: {self.w:.4f}')
            
            global_best_state = self.global_best_model.state_dict()
            for particle in self.swarm:
                particle.update_velocity(global_best_state, self.w, self.c1, self.c2)
                particle.update_position()
            
            if iteration % 10 == 0:
                checkpoint_path = f'checkpoint_iter_{iteration}.pth'
                torch.save(self.global_best_model.state_dict(), checkpoint_path)
                print(f'Checkpoint saved: {checkpoint_path}')
        
        print('PSO optimization completed.')
        print(f'Best Global Score Achieved: {self.global_best_score:.4f}')
        writer.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, marker='o')
        plt.title('PSO Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Global Best Dice Score')
        plt.grid(True)
        plt.savefig('pso_fitness_history.png')
        plt.show()
    
    def get_best_model(self) -> torch.nn.Module:
        """
        Retrieve the best model found during optimization.
        
        Returns:
            torch.nn.Module: The best-performing model.
        """
        return self.global_best_model



def main(model,data_loader):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    model = model.to(device)
    
    pso = PSO(
        model=model,
        num_particles=20,  
        fitness_func=fitness_function,
        data_loader=data_loader,
        max_iterations=100,
        w=0.9,
        c1=2.0,
        c2=2.0,
        verbosity=2, 
        early_stopping_rounds=15
    )
    
    pso.step()
    
    best_model = pso.get_best_model().to(device)
    
    torch.save(best_model.state_dict(), 'with_pso_model.pth')
    print('Best model saved as with_pso_model.pth')
    
    validation_score = fitness_function(best_model, data_loader)
    print(f'Validation Dice Score: {validation_score:.4f}')
    
    pro_prediction(best_model, data_loader)



if __name__ == '__main__':
    set_start_method('spawn')
    main(model, annotated_loader)

    
#cloner174