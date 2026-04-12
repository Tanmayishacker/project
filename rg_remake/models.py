# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Rg Remake Environment.

The Rg Remake environment integrates the Reasoning Gym library to provide
single-step reasoning tasks. Each episode presents one question, the agent submits
an answer, and receives a score.
"""

from typing import Any, Dict,Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class RgRemakeAction(Action):
    """Action for the Rg Remake environment - just a message to echo."""

    answer_msg: str = Field(..., description="Message to echo back")


class RgRemakeObservation(Observation):
    """Observation from the Rg Remake environment - the echoed message."""

    question: Optional[str] = Field(default=None, description="The question to answer (None after step)")

    score: Optional[float] = Field(default=None, description="The score for the answer (0.0 to 1.00, from the dataset.score_answer() )")

    correct_answer: Optional[str] = Field(default=None, description="The correct answer (is revealed after step)")

    dataser_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata from the gym dataset entry")
