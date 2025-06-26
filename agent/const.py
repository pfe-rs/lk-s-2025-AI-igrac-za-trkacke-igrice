max_car_mass=100

max_car_length=200
max_car_width=100

max_car_ni=200
max_car_k=120

max_car_pull=10000

max_car_vx=1000
max_car_vy=1000
max_velocity_scalar = 1400

actions = 4

# for model input
rays_count = 7
params_count = 6
states_count = 4
# rays count doubled cuz there are rays in front of the car
# + in the direction where car moveds
inputs_count = params_count + states_count + 2 * rays_count
