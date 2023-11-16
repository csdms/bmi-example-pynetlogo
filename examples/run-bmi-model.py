import numpy as np

from heat import BmiHeatDiffusion

CONFIG_FILE = "config.yaml"


# Instantiate a model and get its name.
m = BmiHeatDiffusion()
print(m.get_component_name())

# Initialize the model.
m.initialize(CONFIG_FILE)
print(CONFIG_FILE)

# List the model's exchange items.
print("Number of input vars:", m.get_input_item_count())
for var in m.get_input_var_names():
    print(f" - {var}")
print("Number of output vars:", m.get_output_item_count())
for var in m.get_output_var_names():
    print(f" - {var}")

# Get variable info.
var_name = m.get_output_var_names()[0]
print(f"Variable {var_name}")
print(" - variable type:", m.get_var_type(var_name))
print(" - units:", m.get_var_units(var_name))
print(" - itemsize:", m.get_var_itemsize(var_name))
print(" - nbytes:", m.get_var_nbytes(var_name))
print(" - location:", m.get_var_location(var_name))

# Get grid info for variable.
grid_id = m.get_var_grid(var_name)
print(" - grid id:", grid_id)
print(" - grid type:", m.get_grid_type(grid_id))
grid_rank = m.get_grid_rank(grid_id)
print(" - rank:", grid_rank)
grid_size = m.get_grid_size(grid_id)
print(" - size:", grid_size)
grid_shape = np.empty(grid_rank, dtype=np.int32)
try:
    m.get_grid_shape(grid_id, grid_shape)
except RuntimeError:
    print(" - shape: n/a")
else:
    print(" - shape:", grid_shape)
grid_spacing = np.empty(grid_rank, dtype=np.float64)
try:
    m.get_grid_spacing(grid_id, grid_spacing)
except RuntimeError:
    print(" - spacing: n/a")
else:
    print(" - spacing:", grid_spacing)
grid_origin = np.empty(grid_rank, dtype=np.float64)
try:
    m.get_grid_origin(grid_id, grid_origin)
except RuntimeError:
    print(" - origin: n/a")
else:
    print(" - origin:", grid_origin)

# Get time information from the model.
print("Start time:", m.get_start_time())
print("End time:", m.get_end_time())
print("Current time:", m.get_current_time())
print("Time step:", m.get_time_step())
print("Time units:", m.get_time_units())

# Get the variable values.
print(f"Get values of {var_name}...")
val = np.empty(grid_size, dtype=m.get_var_type(var_name))
m.get_value(var_name, val)
print(f" - values at time {m.get_current_time()}, plus sum:")
print(val, val.sum())

# Advance the model by one time step.
print("Advance model by a single time step...")
m.update()
print(" - new time:", m.get_current_time())

# Get the variable values.
print("Get values of {}...".format(var_name))
val = np.empty(grid_size, dtype=m.get_var_type(var_name))
m.get_value(var_name, val)
print(f" - values at time {m.get_current_time()}, plus sum:")
print(val, val.sum())

# Advance the model until a later time.
print("Advance model to a later time...")
while m.get_current_time() < 2.0:
    m.update()
print(" - new time:", m.get_current_time())

# Get the variable values.
print("Get values of {}...".format(var_name))
val = np.empty(grid_size, dtype=m.get_var_type(var_name))
m.get_value(var_name, val)
print(f" - values at time {m.get_current_time()}, plus sum:")
print(val, val.sum())

# Set new variable values.
print(f"Set values of {var_name}...")
new = np.arange(grid_size, dtype=m.get_var_type(var_name))
print(" - values to set:", new[:10])
m.set_value(var_name, new)
val = np.empty(grid_size, dtype=m.get_var_type(var_name))
m.get_value(var_name, val)
print(" - check that values were set:", val[:10])

# Finalize the model.
m.finalize()
print("Done.")
