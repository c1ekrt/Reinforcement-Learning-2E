# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:54:00 2024

@author: jim
"""

from Example13_1_MC_Policy_Gradient import ShortCorridor as MCPG
from Example13_1_MC_Policy_Gradient_with_Baseline import ShortCorridorBias as MCPG_Baseline
import matplotlib.pyplot as plt

mcpg = MCPG()
mcpgbl = MCPG_Baseline()
mcpg.run(500)
mcpgbl.run(500)
plt.legend()
plt.show()