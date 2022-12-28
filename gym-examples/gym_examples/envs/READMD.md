# grid_world.py
### functions introduction
- `def reset()`: resrt env
> For random target and agent location
- `def step(action)`: 
> 1. Cauculate next state  
> 2. Define pad_map, and target move each direction randomly    
> 3. Reward function:
>    reward = 200 if terminated else 0  
>    reward = -50  if die else reward   
