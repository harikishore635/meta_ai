from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import PrescriptionAction, WardObservation

class AntibioticClient(EnvClient[PrescriptionAction, WardObservation, State]):
    """
    Client for the Antibiotic Stewardship Environment.
    """

    def _step_payload(self, action: PrescriptionAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[WardObservation]:
        return StepResult(
            observation=WardObservation.model_validate(payload.get("observation", {})),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {})
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", "default"),
            step_count=payload.get("step_count", 0),
        )
