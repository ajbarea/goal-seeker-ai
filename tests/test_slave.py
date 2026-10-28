"""The robot controller, run against a stand-in for Webots' `controller` module."""

import sys
import types

import pytest


class FakeDevice:
    """Any Webots device: every method returns 0, the compass points north."""

    def getValues(self):
        return [0.0, 0.0, 1.0]

    def __getattr__(self, name):
        return lambda *args, **kwargs: 0


class FakeReceiver(FakeDevice):
    def __init__(self, messages):
        self.messages = list(messages)

    def getQueueLength(self):
        return len(self.messages)

    def getString(self):
        return self.messages[0]

    def nextPacket(self):
        self.messages.pop(0)


class FakeRobot:
    """Steps until the receiver is empty, then ends the simulation."""

    messages: list[str] = []

    def __init__(self):
        self.receiver_device = FakeReceiver(self.messages)

    def getName(self):
        return "robot1"

    def getBasicTimeStep(self):
        return 64

    def getCustomData(self):
        return ""

    def getTime(self):
        return 0.0

    def getDevice(self, name):
        return self.receiver_device if name == "receiver" else FakeDevice()

    def step(self, _time_step):
        return 0 if self.receiver_device.getQueueLength() else -1


@pytest.fixture
def slave_module(monkeypatch):
    fake = types.ModuleType("controller")
    fake.Robot = FakeRobot
    fake.AnsiCodes = types.SimpleNamespace(RED_FOREGROUND="", RESET="")
    monkeypatch.setitem(sys.modules, "controller", fake)
    monkeypatch.delitem(sys.modules, "controllers.slave.slave", raising=False)
    import controllers.slave.slave as slave

    return slave


def test_each_agent_saves_to_its_own_file(slave_module, monkeypatch, tmp_path):
    from controllers.slave.dqn_agent import DQNAgent
    from controllers.slave.q_learning_agent import QLearningAgent

    monkeypatch.setattr(QLearningAgent, "model_path", str(tmp_path / "q.pkl"))
    monkeypatch.setattr(DQNAgent, "model_path", str(tmp_path / "dqn.pt"))
    FakeRobot.messages = [
        "training_mode:dqn",
        "save_q_table",
        "training_mode:q_learning",
        "save_q_table",
    ]
    slave_module.Slave().run()

    assert QLearningAgent().load_q_table(str(tmp_path / "q.pkl"))
    assert DQNAgent(state_dim=5, action_dim=5, device="cpu").load_q_table(
        str(tmp_path / "dqn.pt")
    )
