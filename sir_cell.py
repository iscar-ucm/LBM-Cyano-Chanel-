from __future__ import annotations
from typing import Callable, Any
from xdevs.celldevs.cell import S
from xdevs.celldevs.grid import C, GridCell, GridCellConfig
from xdevs.abc.transducer import Transducible, T
from pandas import DataFrame, read_pickle
from numpy import exp,pi,sin,log
from math import e as math_e

class State(Transducible):
    def __init__(self, conC: float, nutC: float, temp: float, sun: float, upC: float, downC: float):
        self.cC: int = conC
        self.nC: float = nutC
        self.sun: float = sun
        self.temp = temp
        self.upC = upC
        self.downC = downC

    #def __eq__(self, other: State):
    #    return self.nC == other.nC and self.nC == other.nC

    @classmethod
    def transducer_map(cls) -> dict[str, tuple[type[T], Callable[[Any], T]]]:
        return {
            'cyanoC': (float, lambda x: x.cC),
            'upC': (float, lambda x: x.upC),
            'downC': (float, lambda x: x.downC),
            'nutC': (float, lambda x: x.nC),
            'sun': (float, lambda x: x.sun),
            'T': (float, lambda x: x.temp)
        }

class Vicinity:
    def __init__(self, flow_abs: float, inout: int):
        self.flow_abs: float = flow_abs # flow as defined by data colection
        self.inout: float = inout # 1: flow_abs>0 definition is inflow
                                  #-1: flow_abs>0 definition is outflow
        
    @property
    def inflow(self) -> float: 
        return self.flow_abs * self.inout #>0 if flow goes into the cell


class Config:
    def __init__(self, sun0: float, vol: float, dt: float):
        self.sun0 = sun0 # starting day time 
        self.vol = vol
        self.dt = dt  # Seconds between flow data files


class SIRGridCell(GridCell[State, Vicinity]):
    def __init__(self, cell_id: C, config: GridCellConfig):
        super().__init__(cell_id, config)
        self.config: Config = Config(**config.cell_config)
        self.cell_state.sun = self.sunlight()
        if self.cell_id[2] <= 2:
            self.cell_state.temp = 10
        #Neighboring flow initialization
        raw_flowdata = read_pickle('flowdata.pkl')
        for neighbor in self.neighborhood.keys():
            neighbor_col = [col for col in raw_flowdata.columns if all([any(self.cell_id == i for i in col),
                                                                        any(neighbor == i for i in col)])]  
            self.neighborhood[neighbor].flow_abs = raw_flowdata[neighbor_col[0]].values
        self.over_cell = self.cell_id[:2]+(self.cell_id[2]+1,)
        self.under_cell = self.cell_id[:2]+(self.cell_id[2]-1,)

    def local_computation(self, cell_state: S) -> S:
        # 1º Calcular nuevas concentraciones (previo + vmigration + inflow) [cyano, nut]
        # 2º Calcular crecimiento (conc_nutrientes, sol) [Depredador-presa]
        #   dC/dt=a*C-b*C*f(nut,sol)
        #   dnut/dt=-c*nut+d*f(nut,sol)*C
        # 3º Calcular mov vertical (capa_z,sol?,t?)
        time = round(self._clock/self.config.dt)
        new_concC, new_concNut = self.new_conc(cell_state, time)
        cell_state.cC = new_concC
        cell_state.nC = new_concNut
        new_concC, new_concNut = self.cyano_growth(cell_state)
        cell_state.cC = new_concC
        cell_state.nC = new_concNut
        cell_state.upC, cell_state.downC = self.vertical_migration(cell_state)
        cell_state.cC -= (cell_state.upC + cell_state.downC)/self.config.vol
        cell_state.sun = self.sunlight()
        return cell_state

    def new_conc(self, state: State, time: int) -> float:
        neighbor_effect_C = 0
        neighbor_effect_N = 0
        # Vertical migration towards the cell
        if self.over_cell in self.neighborhood.keys():
            state.cC += self.neighbors_state[self.over_cell].downC/self.config.vol
        if self.under_cell in self.neighborhood.keys():
            state.cC += self.neighbors_state[self.under_cell].upC/self.config.vol
        #Flow transportation
        if self.cell_id[:2] != (0,0):
            for neighbor, n_state in self.neighbors_state.items():
                flow = self.neighborhood[neighbor].inflow[time]
                if  flow > 0:
                    neighbor_effect_C += n_state.cC * flow * self.output_delay(state) / self.config.vol
                    neighbor_effect_N += n_state.nC * flow * self.output_delay(state) / self.config.vol
                else:
                    neighbor_effect_C += state.cC * flow * self.output_delay(state) / self.config.vol
                    neighbor_effect_N += state.nC * flow * self.output_delay(state) / self.config.vol

        new_concC =  max(state.cC + neighbor_effect_C,0)
        new_concNut = max(state.nC + neighbor_effect_N,0)
        return new_concC, new_concNut

    def cyano_growth(self, state: State) -> float:
        nA = 0.005; nB=0.000005; 
        cA=0.0019; cB=0.000025; optimalI=150; i = math_e/optimalI
        if self.cell_id[:2] == (0,0):
            state.nC += nA*state.nC
        dn = -nB*state.cC*state.nC

        cI = (i*state.sun*exp(-state.sun/optimalI)) # [0,1] equal to 1 in optimalI
        dC = -cA*state.cC + cB*state.cC*state.nC*(cI*0+1)
        
        new_concC = max(state.cC + dC*self.output_delay(state), 0)
        new_concN = max(state.nC + dn*self.output_delay(state), 0)
        return new_concC, new_concN

    def vertical_migration(self,state: State) -> float:
        # Ascending concentration: (upC in [0,1])
        #   +uA: proportional to sun radiation upwards difference max(0,dsun)
        #   -uB: inversely prop. to conc. gradient
        uA = 0.16; uB = 0.01; 
        # Descending concentration: (upC in [0,1])
        #   +dA: constant decay
        #   +dB: inversely prop to conc. gradient
        #   -dC: inversely prop. to temperature gradient
        dA = 0.2; dB = 0.004; dC = 10
        # Both moving concentrations are limited to a % of current cell cyanobacteria
        uD = 0.25; dD = 0.25

        upC = 0; downC = 0
        if self.over_cell in self.neighborhood.keys():
            upC = uA * log(self.neighbors_state[self.over_cell].sun - state.sun) - \
                uB * max(self.neighbors_state[self.over_cell].cC - state.cC,0)
            upC = max(0,min(upC,1))
            upC = upC * (state.cC * self.config.vol) * uD
        if self.under_cell in self.neighborhood.keys():
            downC = dA - dC * (state.temp - self.neighbors_state[self.under_cell].temp) -\
                dB * max(self.neighbors_state[self.under_cell].cC - state.cC,0)
            downC = max(0, min(downC,1))
            downC = downC * (state.cC * self.config.vol) * dD
        return upC, downC

    def sunlight(self) -> float:
        Ks = 800; K_abs = 0.3; I0 = 10
        sunrise = 6.5 * 3600 # Sunrise at 6:30am --> sunset 12h later
        brightime = 12 # number of light hours in a day [sunrise -> sunset]
        Isurf = Ks * sin(pi/(3600*brightime)*(self._clock + self.config.sun0*3600 - sunrise)) # Irradiance at water surface
        Isurf= I0 + max(0,Isurf)
        I=Isurf*exp(K_abs*-(27-self.cell_id[2]*3)) # Absorption of light
        return I    #i.reshape(len(z),1)

    def output_delay(self, cell_state: S) -> float:
        return self.config.dt/5
