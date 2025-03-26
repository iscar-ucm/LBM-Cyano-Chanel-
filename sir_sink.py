from typing import Dict, NoReturn, Optional, Tuple
from sir_cell import State
from xdevs.celldevs.inout import CellMessage
from xdevs.models import Atomic, Port


class SIRSink(Atomic):
    def __init__(self, name: str = None):
        super().__init__(name)
        self.started: bool = False
        self.cell_reports: Dict[Tuple[int, ...], State] = dict()
        self.scenario_report: Optional[State] = None

        self.in_sink: Port[CellMessage[Tuple[int, ...], State]] = Port(CellMessage, 'in_sink')
        self.out_sink: Port[State] = Port(State, 'out_sink')
        self.add_in_port(self.in_sink)
        self.add_out_port(self.out_sink)

    def deltint(self) -> NoReturn:
        self.passivate()
        self.started = True

    def deltext(self, e: float) -> NoReturn:
        self.activate()
        for msg in self.in_sink.values:
            if self.started:
                prev_report = self.cell_reports[msg.cell_id]
                delta_C = msg.cell_state.cC- prev_report.cC
                delta_N = msg.cell_state.nC - prev_report.nC
                self.scenario_report.cC += delta_C
                self.scenario_report.nC += delta_N
            self.cell_reports[msg.cell_id] = msg.cell_state
        if not self.started:
            self.scenario_report: State = State(0, 0, 0, 0, 0, 0)
            for cell_state in self.cell_reports.values():
                self.scenario_report.cC += cell_state.nC
                self.scenario_report.cC += cell_state.nC

    def lambdaf(self) -> NoReturn:
        self.out_sink.add(self.scenario_report)

    def initialize(self) -> NoReturn:
        self.passivate()

    def exit(self) -> NoReturn:
        pass
