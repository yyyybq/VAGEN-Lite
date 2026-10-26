"""P0 regression: inspect the final policy request, not just reset observations."""
import ast
import asyncio
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.utils.observation_history import ObservationHistory
from vagen.utils.task_context import assert_task_text, check_policy_tokens

TASK = "Position where chair appears to the left of table"
ROOT = Path(__file__).resolve().parents[2]


class InlineExecutor:
    async def run_in_executor(self, executor, function):
        return function()


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("\n".join(str(m['content']) for m in messages))


def env_fixture():
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(render_backend=None, enable_collision_detection=False))
    env.current_item = {"task_description": TASK, "task_type": "projective_relations"}
    env.view_engine.reset(np.eye(4))
    env._render_image = lambda: Image.new("RGB", (32, 32), (50, 100, 150))
    return env


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("window", [1, 3])
def test_task_survives_window_and_both_render_paths(external, window):
    env = env_fixture()
    history = ObservationHistory(window)
    for turn in range(8):
        kwargs = {"init_obs": turn == 0, "task_prompt": TASK if turn == 0 else ""}
        obs = env._build_observation_from_image(Image.new("RGB", (32, 32)), **kwargs) if external else env._render(**kwargs)
        history.append({"role": "user", "content": obs["obs_str"]}, [turn])
        assert_task_text(history.messages[-1]["content"], TASK)
        while history.drop_oldest_turn():
            pass
        ids = Tokenizer().apply_chat_template(history.messages)
        assert check_policy_tokens(Tokenizer(), ids, TASK, len(ids))["task_present"]
        history.append({"role": "assistant", "content": "<action>turn_left|</action>"})
    assert history.images == [7]


def loop_method(name):
    tree = ast.parse((ROOT / 'vagen/agent_loop/gym_agent_loop_no_concat.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GymAgentLoop')
    fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    namespace = {"simple_timer": lambda *a: nullcontext(),
                 "AgentState": SimpleNamespace(INTERACTING="done", GENERATING="generate"),
                 "_flatten_text_only_content": lambda m: m}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-loop-method>', 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize("mode", ["valid", "lost_task", "over_budget"])
def test_exact_inference_boundary(mode):
    async def run():
        tokenizer = Tokenizer()
        ids = tokenizer.encode('Task: ' + TASK if mode != 'lost_task' else '[Observation]: image')
        calls = []
        async def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(token_ids=tokenizer.encode('<action>turn_left|</action>'), log_probs=[])
        loop = SimpleNamespace(tokenizer=tokenizer, loop=InlineExecutor(), response_length=384,
                               prompt_length=len(ids)-(1 if mode=='over_budget' else 0),
                               server_manager=SimpleNamespace(generate=generate))
        data = SimpleNamespace(public_task=TASK, turn_prompt_ids=ids, response_limit=384,
                               history=ObservationHistory(1), sys_images=[], metrics={}, request_id='test')
        fn = loop_method('_handle_generating_state')
        if mode == 'valid':
            await fn(loop, data, {})
            assert_task_text(tokenizer.decode(calls[0]['prompt_ids'][:len(ids)]), TASK)
            assert data.task_context_check['task_present']
        else:
            with pytest.raises(ValueError):
                await fn(loop, data, {})
            assert not calls
    asyncio.run(run())


def test_pending_drops_whole_old_turns_not_current_task():
    async def run():
        history = ObservationHistory(3)
        history.append({'role':'user','content':'old '*1000})
        history.append({'role':'assistant','content':'move'})
        history.append({'role':'user','content':'Task: '+TASK})
        data = SimpleNamespace(history=history, sys_images=[], sys_msg={'role':'system','content':'system'})
        fn = loop_method('_handle_pending_state')
        loop = SimpleNamespace(processor=None, tokenizer=Tokenizer(), loop=InlineExecutor(),
                               apply_chat_template_kwargs={}, prompt_length=300)
        async def pending(d,s):return await fn(loop,d,s)
        loop._handle_pending_state = pending
        await fn(loop,data,{})
        assert len(history.messages)==1
        check_policy_tokens(loop.tokenizer,data.turn_prompt_ids,TASK,300)
    asyncio.run(run())


def test_evaluation_agent_uses_same_task_contract(monkeypatch):
    from evaluation.model_agent import ModelAgent
    from evaluation.eval_config import EvalModelConfig
    monkeypatch.setattr(ModelAgent, '_init_model', lambda self: None)
    agent = ModelAgent(EvalModelConfig(provider='vllm'), history_window_size=1)
    sent = []
    agent._generate_vllm = lambda images: sent.append(agent.conversation_history[-1]['content']) or '<action>turn_left|</action>'
    for i in range(5):
        agent.act({'obs_str': f'Task: {TASK}\nframe {i}'})
    assert all(TASK in text for text in sent)
    with pytest.raises(ValueError):
        agent.act({'obs_str': 'frame without task'})


def test_sft_subsequent_observations_retain_task():
    from data_gen.active_spatial_sft.path_finder import Trajectory, TrajStep
    from data_gen.active_spatial_sft.sft_formatter import format_trajectory
    pose = np.eye(4)
    step = TrajStep(0, pose, ['turn_left'], pose, .1, .2, .1, .2, .1, .2)
    step_2 = TrajStep(1, pose, ['turn_right'], pose, .2, .3, .2, .3, .2, .3)
    trajectory = Trajectory([step, step_2], pose, .1, .3, False, 2)
    record = format_trajectory({'task_description': TASK, 'task_type': 'projective_relations'},
                               trajectory, ['initial.png', 'next.png', 'final.png'], 'task-test', enable_explicit_done=False)
    users = [m['content'] for m in record['conversations'] if m['role'] == 'user']
    assert len(users) == 2
    for text in users:
        assert_task_text(text, TASK)


def test_evaluation_checks_backend_actual_prompt_tokens(monkeypatch):
    from evaluation.model_agent import ModelAgent
    from evaluation.eval_config import EvalModelConfig
    monkeypatch.setattr(ModelAgent, '_init_model', lambda self: None)
    agent = ModelAgent(EvalModelConfig(provider='vllm'))
    agent.public_task = TASK
    agent.model = SimpleNamespace(get_tokenizer=lambda: Tokenizer(),
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)))
    agent._validate_vllm_input(SimpleNamespace(prompt_token_ids=Tokenizer().encode('Task: '+TASK)))
    assert agent.last_task_context_check['task_present']
    with pytest.raises(ValueError, match='lost its public task'):
        agent._validate_vllm_input(SimpleNamespace(prompt_token_ids=Tokenizer().encode('missing task')))


@pytest.mark.skipif(not os.environ.get('R1_TEST_MODEL_PATH'), reason='requires local Qwen model processor')
def test_real_qwen_image_expansion_and_budget_eviction():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(os.environ['R1_TEST_MODEL_PATH'], local_files_only=True)
    async def run():
        history = ObservationHistory(3)
        im = Image.new('RGB', (256, 256))
        def message(text):
            return {'role': 'user', 'content': [{'type': 'image', 'image': im}, {'type': 'text', 'text': text}]}
        history.append(message('old context ' * 6000), [im])
        history.append({'role': 'assistant', 'content': '<action>turn_left|</action>'})
        history.append(message('Task: ' + TASK), [im])
        data = SimpleNamespace(history=history, sys_images=[], sys_msg={'role': 'system', 'content': 'Navigate.'})
        fn = loop_method('_handle_pending_state')
        loop = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer, loop=InlineExecutor(),
                               apply_chat_template_kwargs={}, prompt_length=4096)
        async def pending(d, s):
            return await fn(loop, d, s)
        loop._handle_pending_state = pending
        await fn(loop, data, {})
        assert len(history.messages) == len(history.images) == 1
        check_policy_tokens(processor.tokenizer, data.turn_prompt_ids, TASK, 4096)
        # Same actual image-expanded prompt cannot be silently sliced smaller.
        loop.prompt_length = len(data.turn_prompt_ids) - 1
        with pytest.raises(ValueError, match='current observation exceeds'):
            await fn(loop, data, {})
    asyncio.run(run())
