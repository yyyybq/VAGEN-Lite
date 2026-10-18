"""Transition questions for InteriorGS v10 SFT states (docs/transition_qa/PROGRESS.md).

Three question classes, all coupled to the task of the trajectory they come from:

  * state    - what the current image shows about the task's objects (is the task met, left / right, which is
               closer, how far, how hidden, both in view, how tall on screen);
  * forward  - the same quantities after one action of the action space repeated k times (no rendering);
  * inverse  - which action leads from one SFT frame to the next one (two images, no task text).

Every answer comes from the env's or the generator's own functions (geometry.py); nothing is re-implemented.
"""
