#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``ACTConfig.action_delta_offset``: shifting the sampled action window.

Mirrors ``tests/policies/groot/test_groot_action_delta_offset.py``. The two policies must agree on
the delta geometry -- a dataset whose ``action[t] == state[t]`` needs the same shift whichever of
them is trained on it -- so the window assertions here are deliberately the same shape. The tail
handling is where they legitimately differ; see ``test_drop_n_last_frames_*`` below.
"""

import pytest

from lerobot.policies.act.configuration_act import ACTConfig


def _config(**kwargs) -> ACTConfig:
    return ACTConfig(device="cpu", **kwargs)


def test_default_offset_is_backward_compatible():
    """Offset 0 must reproduce upstream ACT exactly, tail handling included."""
    config = _config(chunk_size=100, n_action_steps=100)

    assert config.action_delta_indices == list(range(100))
    assert config.drop_n_last_frames == 0


def test_offset_shifts_the_window_without_changing_its_length():
    config = _config(chunk_size=50, n_action_steps=50, action_delta_offset=1)

    assert config.action_delta_indices == list(range(1, 51))
    assert len(config.action_delta_indices) == 50


def test_drop_n_last_frames_follows_the_offset():
    """Exactly the frames whose whole window is out of bounds, and no more.

    At offset k the last k frames have no in-bounds action step, so their L1 loss is entirely
    masked out. Everything before them keeps at least one real step, which is the invariant
    upstream ACT already relies on.
    """
    config = _config(chunk_size=50, n_action_steps=50, action_delta_offset=1)

    assert config.drop_n_last_frames == 1

    config = _config(chunk_size=50, n_action_steps=50, action_delta_offset=4)

    assert config.drop_n_last_frames == 4


def test_drop_n_last_frames_does_not_follow_the_largest_index():
    """The GR00T formulation would discard most of every episode here.

    GR00T keys its drop off ``max(action_delta_indices)`` because it has no padding mask and needs a
    complete chunk. ACT masks padded steps out of the loss, so copying that would throw away
    ``chunk_size - 1`` tail frames for no benefit -- 49 of a ~105-frame episode at this chunk size.
    """
    config = _config(chunk_size=50, n_action_steps=50, action_delta_offset=1)

    assert config.drop_n_last_frames < max(config.action_delta_indices)


@pytest.mark.parametrize("offset", [0, 1, 3])
def test_offset_never_shortens_the_chunk(offset):
    config = _config(chunk_size=8, n_action_steps=8, action_delta_offset=offset)

    assert len(config.action_delta_indices) == 8
    assert config.action_delta_indices[0] == offset


def test_negative_offset_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        _config(chunk_size=50, n_action_steps=50, action_delta_offset=-1)


def test_observation_delta_indices_are_unaffected():
    """The observation stays at t; only the action window moves."""
    config = _config(chunk_size=50, n_action_steps=50, action_delta_offset=1)

    assert config.observation_delta_indices is None
