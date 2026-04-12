# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rg Remake Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import RgRemakeAction, RgRemakeObservation


class RgRemakeEnv(
    EnvClient[RgRemakeAction, RgRemakeObservation, State]
):
    """
    Client for the Rg Remake Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> from rg_remake import RgRemakeAction, RgRemakeObservation
        >>>
        >>> env = RgRemakeAction(base_url="http://localhost:8000")
        >>> # Create dataset with 10 leg_counting questions
        >>> result = env.reset(
        ...     dataset_name='leg_counting',
        ...     dataset_config={"min_animals": 5, "max_animals": 15},
        ...     seed=42,
        ...     size=10
        ... )
        >>> print(f"Question: {result.observation.question}")
        >>>
        >>> # Answer question
        >>> result = env.step(RgRemakeObservation(answer="4"))
        >>> print(f"Score: {result.observation.score}")
        >>> print(f"Correct: {result.observation.correct_answer}")
        >>>
        >>> # Get next question (reuses dataset)
        >>> result = env.reset()
        >>> print(f"Next question: {result.observation.question}")
        >>> env.close()

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = rg_remake.from_docker_image("rg_remake:latest")
        >>> try:
        ...     result = client.reset(
        ...         dataset_name='leg_counting',
        ...         dataset_config={"min_animals": 5, "max_animals": 15},
        ...         seed=42,
        ...         size=5
        ...     )
        ...     result = client.step(RgRemakeAction(answer="4"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RgRemakeAction) -> Dict:
        """
        Convert RgRemakeAction to JSON payload for step message.

        Args:
            action: RgRemakeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "answer_msg": action.answer_msg,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RgRemakeObservation]:
        """
        Parse server response into StepResult[RgRemakeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with RgRemakeObservation
        """
        obs_data = payload.get("observation", {})
        observation = RgRemakeObservation(
            question=obs_data.get("question"),
            score=obs_data.get("score"),
            correct_answer=obs_data.get("correct_answer"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
            dataset_metadata=obs_data.get("dataset_metadata"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
