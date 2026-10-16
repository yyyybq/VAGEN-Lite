"""CPU checks of reachable extension boundaries and separate occlusion drafts."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from data_gen.interiorgs_qa.capability import extend
from data_gen.interiorgs_v10.contract import apply_action, pose_from_xy_yaw


class FakeScene:
    min_wall_distance = .5

    def __init__(self, wall_distance=1., block_transition=False):
        self.distance, self.block = wall_distance, block_transition
        self.detector = SimpleNamespace(check_collision=lambda p: SimpleNamespace(has_collision=False))

    def room_of(self, x, y):
        return 0 if abs(x) < 10 and abs(y) < 10 else None

    def wall_distance(self, x, y):
        return self.distance

    def transition_legal(self, a, b):
        return (False, 'collision_wall') if self.block else (True, 'ok')


def occlusion_measurement():
    def obj(visible):
        return {'visibility': {'visible': visible}, 'references': [],
                'projection': {'all_corners_in_front': True, 'inside_fraction': 1.}}
    return {'task_type': 'occlusion_alignment', 'objects': [obj(False), obj(True)],
            'metric': {'success': True, 'target_hidden_fraction': .98}}


class ExtensionTests(unittest.TestCase):
    def test_source_prefix_is_replayed_and_pose_mismatch_rejected(self):
        initial = pose_from_xy_yaw(0, 0, 0)
        actions = ['move_forward', 'turn_left']
        pose = initial
        for action in actions:
            pose = apply_action(pose, action)
        task = {'init_camera': {'extrinsics': initial.tolist()}}
        state = {'actions_from_initial': actions, 'c2w': pose.tolist()}
        actual, result = extend.replay_seed(task, state, FakeScene())
        np.testing.assert_allclose(actual, pose)
        self.assertTrue(result['sample_eligible'])
        wrong = copy.deepcopy(state)
        wrong['c2w'][0][3] += .3
        with self.assertRaises(ValueError):
            extend.replay_seed(task, wrong, FakeScene())

    def test_wall_room_and_transition_checks_are_not_bypassed(self):
        pose = pose_from_xy_yaw(0, 0, 0)
        moved = apply_action(pose, 'move_forward')
        self.assertFalse(extend.check_step(FakeScene(.49), pose, moved)['sample_eligible'])
        self.assertFalse(extend.check_step(FakeScene(block_transition=True), pose, moved)['runtime_legal'])
        outside = pose_from_xy_yaw(20, 0, 0)
        self.assertFalse(extend.check_step(FakeScene(), pose, outside)['runtime_legal'])

    def test_hidden_target_can_be_render_priority_without_single_image_visibility(self):
        measured = occlusion_measurement()
        candidate = extend.occlusion_geometry_candidate(measured)
        self.assertEqual(candidate['answer_key'], 'fully_hidden')
        self.assertTrue(candidate['geometry_only'])
        self.assertFalse(candidate['final_eligible'])
        self.assertTrue(candidate['needs_rendered_depth'])
        measured['metric']['success'] = False
        self.assertIsNone(extend.occlusion_geometry_candidate(measured))
        measured['objects'][0]['projection']['inside_fraction'] = 0.
        self.assertEqual(extend.occlusion_geometry_candidate(measured)['answer_key'], 'outside')
        measured['objects'][1]['visibility']['visible'] = False
        self.assertIsNone(extend.occlusion_geometry_candidate(measured))

    def test_identification_must_be_an_earlier_eligible_prefix_of_same_parent(self):
        s = {'state_id': 'before', 'parent_task_id': 'task', 'actions_from_initial': ['move_forward'],
             'sample_eligible': True, 'runtime_legal': True}
        m = occlusion_measurement()
        m['objects'][0]['visibility']['visible'] = True
        g = {'before': {'measurement': m}}
        self.assertEqual(extend.identifying_prefix('task', ['move_forward', 'turn_left'], [s], g), 'before')
        self.assertIsNone(extend.identifying_prefix('task', ['turn_left', 'move_forward'], [s], g))
        self.assertIsNone(extend.identifying_prefix('other', ['move_forward', 'turn_left'], [s], g))
        self.assertIsNone(extend.identifying_prefix('task', ['move_forward'], [s], g))
        s['sample_eligible'] = False
        self.assertIsNone(extend.identifying_prefix('task', ['move_forward', 'turn_left'], [s], g))

    def test_semantic_quota_counts_questions_without_changing_their_answer(self):
        desired = extend.desired_counts(500)
        self.assertEqual(sum(desired.values()), 500)
        self.assertEqual(desired['projective_direction:right'], 25)
        self.assertEqual(desired['occlusion_closer:a_closer'], 25)
        rec = {'drafts': [{'kind': 'occlusion_closer', 'answer_key': 'b_closer'}],
               'occlusion_geometry_candidate': {'answer_key': 'not_hidden'}}
        before = copy.deepcopy(rec)
        self.assertEqual(extend.rec_categories(rec), {'occlusion_closer:b_closer', 'occlusion_state:not_hidden'})
        self.assertEqual(rec, before)

    def test_current_camera_identity_deduplicates_across_parent_sources(self):
        pose = pose_from_xy_yaw(0, 0, 0)
        K = [[128., 0., 128.], [0., 170.6666666667, 128.], [0., 0., 1.]]
        key = extend.view_key('scene', pose, K)
        self.assertEqual(key, extend.view_key('scene', pose.tolist(), K))
        self.assertNotEqual(key, extend.view_key('other_scene', pose, K))


if __name__ == '__main__':
    unittest.main()
