from gym.envs.registration import register

register(
    id='kuka_iiwa_insertion-v0', 
    entry_point='kuka_iiwa_insertion.envs:IiwaInsertionEnv'
)