from __future__ import annotations


class NemoIRRuntimeError(Exception):
    pass


class ToolValidationError(NemoIRRuntimeError):
    pass


class MissingCapabilityError(NemoIRRuntimeError):
    pass


class ToolInvocationError(NemoIRRuntimeError):
    pass


class PolicyDeniedError(NemoIRRuntimeError):
    pass


class PolicyEvaluationError(NemoIRRuntimeError):
    pass


class StageOutputValidationError(NemoIRRuntimeError):
    pass


class DataUnavailableError(NemoIRRuntimeError):
    pass


class NoTransitionMatchedError(NemoIRRuntimeError):
    pass


class MaxStepsExceededError(NemoIRRuntimeError):
    pass


class WorkflowTimeoutError(NemoIRRuntimeError):
    pass


class WorkflowValidationError(NemoIRRuntimeError):
    pass
