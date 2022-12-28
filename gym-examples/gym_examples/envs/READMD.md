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
- `Action`: (up, right, down, left)    
```
self._action_to_direction = {
  0: np.array([1, 0]),
  1: np.array([0, 1]),
  2: np.array([-1, 0]),
  3: np.array([0, -1]),
}
```
