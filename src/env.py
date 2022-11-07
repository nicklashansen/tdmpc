from collections import deque, defaultdict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import gym
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ExtendedTimeStep(NamedTuple):
	step_type: Any
	reward: Any
	discount: Any
	observation: Any
	action: Any

	def first(self):
		return self.step_type == StepType.FIRST

	def mid(self):
		return self.step_type == StepType.MID

	def last(self):
		return self.step_type == StepType.LAST


class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
	def __init__(self, env, num_frames, pixels_key='pixels'):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._pixels_key = pixels_key

		wrapped_obs_spec = env.observation_spec()
		assert pixels_key in wrapped_obs_spec

		pixels_shape = wrapped_obs_spec[pixels_key].shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		self._obs_spec = specs.BoundedArray(shape=np.concatenate(
			[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
											dtype=np.uint8,
											minimum=0,
											maximum=255,
											name='observation')

	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		obs = np.concatenate(list(self._frames), axis=0)
		return time_step._replace(observation=obs)

	def _extract_pixels(self, time_step):
		pixels = time_step.observation[self._pixels_key]
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		return pixels.transpose(2, 0, 1).copy()

	def reset(self):
		time_step = self._env.reset()
		pixels = self._extract_pixels(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
		return self._transform_observation(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		self._frames.append(pixels)
		return self._transform_observation(time_step)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		wrapped_action_spec = env.action_spec()
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')

	def step(self, action):
		action = action.astype(self._env.action_spec().dtype)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
	def __init__(self, env):
		self._env = env

	def reset(self):
		time_step = self._env.reset()
		return self._augment_time_step(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		return self._augment_time_step(time_step, action)

	def _augment_time_step(self, time_step, action=None):
		if action is None:
			action_spec = self.action_spec()
			action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
		return ExtendedTimeStep(observation=time_step.observation,
								step_type=time_step.step_type,
								action=action,
								reward=time_step.reward or 0.0,
								discount=time_step.discount or 1.0)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class TimeStepToGymWrapper(object):
	def __init__(self, env, domain, task, action_repeat, modality):
		try: # pixels
			obs_shp = env.observation_spec().shape
			assert modality == 'pixels'
		except: # state
			obs_shp = []
			for v in env.observation_spec().values():
				try:
					shp = np.prod(v.shape)
				except:
					shp = 1
				obs_shp.append(shp)
			obs_shp = (np.sum(obs_shp, dtype=np.int32),)
			assert modality != 'pixels'
		act_shp = env.action_spec().shape
		obs_dtype = np.float32 if modality != 'pixels' else np.uint8
		self.observation_space = gym.spaces.Box(
			low=np.full(
				obs_shp,
				-np.inf if modality != 'pixels' else env.observation_spec().minimum,
				dtype=obs_dtype),
			high=np.full(
				obs_shp,
				np.inf if modality != 'pixels' else env.observation_spec().maximum,
				dtype=obs_dtype),
			shape=obs_shp,
			dtype=obs_dtype,
		)
		self.action_space = gym.spaces.Box(
			low=np.full(act_shp, env.action_spec().minimum),
			high=np.full(act_shp, env.action_spec().maximum),
			shape=act_shp,
			dtype=env.action_spec().dtype)
		self.env = env
		self.domain = domain
		self.task = task
		self.ep_len = 1000//action_repeat
		self.modality = modality
		self.t = 0
	
	@property
	def unwrapped(self):
		return self.env

	@property
	def reward_range(self):
		return None

	@property
	def metadata(self):
		return None
	
	def _obs_to_array(self, obs):
		if self.modality != 'pixels':
			return np.concatenate([v.flatten() for v in obs.values()])
		return obs

	def reset(self):
		self.t = 0
		return self._obs_to_array(self.env.reset().observation)
	
	def step(self, action):
		self.t += 1
		time_step = self.env.step(action)
		return self._obs_to_array(time_step.observation), time_step.reward, time_step.last() or self.t == self.ep_len, defaultdict(float)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)


class DefaultDictWrapper(gym.Wrapper):
	def __init__(self, env):
		gym.Wrapper.__init__(self, env)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return obs, reward, done, defaultdict(float, info)


def make_env(cfg):
	"""
	Make DMControl environment for TD-MPC experiments.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup').get(domain, domain)
	assert (domain, task) in suite.ALL_TASKS
	env = suite.load(domain,
					 task,
					 task_kwargs={'random': cfg.seed},
					 visualize_reward=False)
	env = ActionDTypeWrapper(env, np.float32)
	env = ActionRepeatWrapper(env, cfg.action_repeat)
	env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

	if cfg.modality=='pixels':
		if (domain, task) in suite.ALL_TASKS:
			camera_id = dict(quadruped=2).get(domain, 0)
			render_kwargs = dict(height=84, width=84, camera_id=camera_id)
			env = pixels.Wrapper(env,
								pixels_only=True,
								render_kwargs=render_kwargs)
		env = FrameStackWrapper(env, cfg.get('frame_stack', 1), cfg.modality)
	env = ExtendedTimeStepWrapper(env)
	env = TimeStepToGymWrapper(env, domain, task, cfg.action_repeat, cfg.modality)
	env = DefaultDictWrapper(env)

	# Convenience
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]

	return env
