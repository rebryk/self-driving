from collections import namedtuple

TrajectoryData = namedtuple("TrajectoryData", [
    'intended_lane',
    'final_lane',
    'end_distance_to_goal',
])

"""
Here we have provided two possible suggestions for cost functions, but feel free to use your own!
The weighted cost over all cost functions is computed in calculate_cost. See get_helper_data
for details on how useful helper data is computed.
"""

# weights for costs
REACH_GOAL = 1
EFFICIENCY = 0.5
TIME = 2
EPS = 1e-2

DEBUG = False


def goal_distance_cost(vehicle, trajectory, predictions, data):
    """
    Cost increases based on distance of intended lane (for planning a lane change) and final lane of a trajectory.
    Cost of being out of goal lane also becomes larger as vehicle approaches goal distance.
    """

    lane_delta = abs(vehicle.goal_lane - data.intended_lane) + abs(vehicle.goal_lane - data.final_lane)
    average_lane_delta = lane_delta / 2
    return average_lane_delta / max(data.end_distance_to_goal, EPS)


def position_at(position, velocity, acceleration, time):
    return position + velocity * time + 0.5 * acceleration * time ** 2


def inefficiency_cost(vehicle, trajectory, predictions, data):
    """
    Cost becomes higher for trajectories with intended lane and final lane that have slower traffic. 
    """

    intended_kinematics = vehicle.get_kinematics(predictions, data.intended_lane)
    intended_distance = max(vehicle.goal_s - position_at(*intended_kinematics, time=TIME), 0)

    final_kinematics = vehicle.get_kinematics(predictions, data.final_lane)
    final_distance = max(vehicle.goal_s - position_at(*final_kinematics, time=TIME), 0)

    average_distance = (intended_distance + final_distance) / 2

    return average_distance / vehicle.goal_s


def calculate_cost(vehicle, trajectory, predictions):
    """
    Sum weighted cost functions to get total cost for trajectory.
    """
    trajectory_data = get_helper_data(vehicle, trajectory, predictions)
    # print(trajectory_data)
    cost = 0.0
    cf_list = [goal_distance_cost, inefficiency_cost]
    weight_list = [REACH_GOAL, EFFICIENCY]

    for weight, cf in zip(weight_list, cf_list):
        new_cost = weight * cf(vehicle, trajectory, predictions, trajectory_data)
        cost += new_cost
    return cost


def get_helper_data(vehicle, trajectory, predictions):
    """
    Generate helper data to use in cost functions:
    indended_lane:  +/- 1 from the current lane if the ehicle is planning or executing a lane change.
    final_lane: The lane of the vehicle at the end of the trajectory. The lane is unchanged for KL and PLCL/PLCR trajectories.
    distance_to_goal: The s distance of the vehicle to the goal.

    Note that indended_lane and final_lane are both included to help differentiate between planning and executing
    a lane change in the cost functions.
    """

    last = trajectory[1]

    if last.state == "PLCL":
        intended_lane = last.lane + 1
    elif last.state == "PLCR":
        intended_lane = last.lane - 1
    else:
        intended_lane = last.lane

    distance_to_goal = vehicle.goal_s - last.s
    final_lane = last.lane

    return TrajectoryData(
        intended_lane,
        final_lane,
        distance_to_goal)


def velocity(predictions, lane):
    """
    All non ego vehicles in a lane have the same speed, so to get the speed limit for a lane,
    we can just find one vehicle in that lane.
    """
    for v_id, predicted_traj in predictions.items():
        if predicted_traj[0].lane == lane and v_id != -1:
            return predicted_traj[0].v
