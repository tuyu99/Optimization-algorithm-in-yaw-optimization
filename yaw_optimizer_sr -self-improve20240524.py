# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import copy
import warnings
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris.logging_manager import LoggerBase

# from .yaw_optimizer_scipy import YawOptimizationScipy
from .yaw_optimization_base import YawOptimization


class YawOptimizationSR(YawOptimization, LoggerBase):
    def __init__(
        self,
        fi,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=2,  # Optimization options
        turbine_weights=None,
        exclude_downstream_turbines=True,
        exploit_layout_symmetry=True,
        verify_convergence=False,
    ):
        """
        Instantiate YawOptimizationSR object with a FlorisInterface object
        and assign parameter values.
        """
        # 20240407改动
        # 初始化时，将计数器初始化为0
        self.num_calls_to_calc_power = 0

        # Initialize base class
        super().__init__(
            fi=fi,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            exploit_layout_symmetry=exploit_layout_symmetry,
            verify_convergence=verify_convergence,
        )

        # Start a timer for FLORIS computations
        self.time_spent_in_floris = 0

        # Save optimization choices to self
        self.Ny_passes = Ny_passes
        

        # For each wind direction, determine the order of turbines
        self._get_turbine_orders()

    def _get_turbine_orders(self):
        layout_x = self.fi.layout_x
        layout_y = self.fi.layout_y
        turbines_ordered_array = []
        for wd in self.fi_subset.floris.flow_field.wind_directions:
            layout_x_rot = (
                np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
                - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
            )
            # print('0. layout_x_rot',layout_x_rot)
            turbines_ordered = np.argsort(layout_x_rot)
            # print('1.turbines_ordered ',turbines_ordered)
            turbines_ordered_array.append(turbines_ordered)
        self.turbines_ordered_array_subset = np.vstack(turbines_ordered_array)


    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        # 20240407改动
        # 在这个方法中，每次调用都增加计数器的值
        self.num_calls_to_calc_power += 1
        
        # Define current optimal solutions and floris wind directions locally
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        farm_power_opt_subset = self._farm_power_opt_subset
        wd_array_subset = self.fi_subset.floris.flow_field.wind_directions
        turbine_weights_subset = self._turbine_weights_subset

        # Reformat yaw_angles_subset, if necessary
        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 4)
        if eval_multiple_passes:
            # Four-dimensional; format everything into three-dimensional
            Ny = yaw_angles_subset.shape[0]  # Number of passes
            yaw_angles_subset = np.vstack(
                [yaw_angles_subset[iii, :, :, :] for iii in range(Ny)]
            )
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1, 1))
            farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny, 1))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1, 1))

        # Initialize empty matrix for floris farm power outputs
        farm_powers = np.zeros((yaw_angles_subset.shape[0], yaw_angles_subset.shape[1]))

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=2).all(axis=1)
            farm_powers[idx, :] = farm_power_opt_subset[idx, :]
            if self.print_progress:
                self.logger.info(
                    "Skipping {:d}/{:d} calculations: already in memory.".format(
                        np.sum(idx), len(idx))
                )
        else:
            idx = np.zeros(yaw_angles_subset.shape[0], dtype=bool)

        if not np.all(idx):
            # Now calculate farm powers for conditions we haven't yet evaluated previously
            start_time = timerpc()
            farm_powers[~idx, :] = self._calculate_farm_power(
                wd_array=wd_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :, :],
                yaw_angles=yaw_angles_subset[~idx, :, :],
            )
            self.time_spent_in_floris += (timerpc() - start_time)

        # Finally format solutions back to original format, if necessary
        if eval_multiple_passes:
            farm_powers = np.reshape(
                farm_powers,
                (
                    Ny,
                    self.fi_subset.floris.flow_field.n_wind_directions,
                    self.fi_subset.floris.flow_field.n_wind_speeds
                )
            )

        return farm_powers

    def _generate_evaluation_grid(self, turbine_depth):
        """
        Calculate the yaw angles for every iteration in the SR algorithm.
        """
        for iw in range(self._nwinddirections_subset):
            turbid = self.turbines_ordered_array_subset[iw, turbine_depth]  # Turbine to manipulate

            # Grab yaw bounds from self
            yaw_lb = self._yaw_lbs[iw, :, turbid]
            yaw_ub = self._yaw_ubs[iw, :, turbid]

            # Saturate to allowable yaw limits
            yaw_lb = np.clip(
                yaw_lb,
                self.minimum_yaw_angle[iw, :, turbid],
                self.maximum_yaw_angle[iw, :, turbid]
            )
            yaw_ub = np.clip(
                yaw_ub,
                self.minimum_yaw_angle[iw, :, turbid],
                self.maximum_yaw_angle[iw, :, turbid]
            )
            Ny = int(yaw_ub[-1] - yaw_lb[-1] + 1)  # 确保Ny是整数
            evaluation_grid = np.tile(self._yaw_angles_opt_subset, (Ny, 1, 1, 1))
            
            yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny)
            evaluation_grid[:, iw, :, turbid] = yaw_angles_subset
            
        self._yaw_evaluation_grid = evaluation_grid
        return evaluation_grid

    def _process_evaluation_grid(self):
        # Evaluate the farm AEPs for the grid of possible yaw angles
        evaluation_grid = self._yaw_evaluation_grid
        farm_powers = self._calc_powers_with_memory(evaluation_grid)
        return farm_powers

    def optimize(self, print_progress=True):
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """
        self.print_progress = print_progress

        # For each pass, from front to back
        ii = 0
        for Nii in range(self.Ny_passes):
            # Disturb yaw angles for one turbine at a time, from front to back
            for turbine_depth in range(self.nturbs):
                p = 100.0 * ii / ((self.Ny_passes) * self.nturbs)
                ii += 1
                if self.print_progress:
                    print(
                        f"[Serial Refine] Processing pass={Nii}, "
                        f"turbine_depth={turbine_depth} ({p:.1f}%)"
                    )

                # Create grid to evaluate yaw angles for one turbine == turbine_depth
                evaluation_grid = self._generate_evaluation_grid(
                    turbine_depth=turbine_depth
                )
                print('1. evaluation_grid',evaluation_grid.shape)
                # Evaluate grid of yaw angles, get farm powers and find optimal solutions
                farm_powers = self._process_evaluation_grid()

                # If farm powers contains any nans, then issue a warning
                if np.any(np.isnan(farm_powers)):
                    err_msg = (
                        "NaNs found in farm powers during SerialRefine "
                        "optimization routine. Proceeding to maximize over yaw "
                        "settings that produce valid powers."
                    )
                    self.logger.warning(err_msg, stack_info=True)

                # Find optimal solutions in new evaluation grid
                args_opt = np.expand_dims(np.nanargmax(farm_powers, axis=0), axis=0)
                farm_powers_opt_new = np.squeeze(
                    np.take_along_axis(farm_powers, args_opt, axis=0),
                    axis=0,
                )
                # print('4. farm_powers_opt_new',farm_powers_opt_new)
                
                yaw_angles_opt_new = np.squeeze(
                    np.take_along_axis(
                        evaluation_grid,
                        np.expand_dims(args_opt, axis=3),
                        axis=0
                    ),
                    axis=0
                )
                # print('5. yaw_angles_opt_new',yaw_angles_opt_new)
                

                farm_powers_opt_prev = self._farm_power_opt_subset
                yaw_angles_opt_prev = self._yaw_angles_opt_subset

                # Now update optimal farm powers if better than previous
                ids_better = (farm_powers_opt_new > farm_powers_opt_prev)
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids_better] = farm_powers_opt_new[ids_better]

                # Now update optimal yaw angles if better than previous
                turbs_sorted = self.turbines_ordered_array_subset
                # print('6. turbs_sorted0',turbs_sorted)
                turbids = turbs_sorted[np.where(ids_better)[0], turbine_depth]
                ids = (*np.where(ids_better), turbids)
                yaw_angles_opt = yaw_angles_opt_prev
                yaw_angles_opt[ids] = yaw_angles_opt_new[ids]

                self._yaw_lbs[ids] = np.clip(
                    np.round(yaw_angles_opt[ids] - np.abs(1 / (Nii + 2) * self._minimum_yaw_angle_subset[ids])),
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )
                self._yaw_ubs[ids] = np.clip(
                    np.round(yaw_angles_opt[ids] + np.abs(1 / (Nii + 2) * self._maximum_yaw_angle_subset[ids])),
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )
                # print('ids',ids)
                # print(f"Nii {Nii}, turbine_depth {turbine_depth}, yaw_angles_opt: {yaw_angles_opt}")
                # print(f"Nii {Nii}, turbine_depth {turbine_depth}, evaluation_grid: {evaluation_grid}")
                # print('evaluation_grid:',type(evaluation_grid),evaluation_grid.shape)
                # print(f"Nii {Nii}, turbine_depth {turbine_depth}, yaw_lbs: {self._yaw_lbs}, yaw_ubs: {self._yaw_ubs}")
                # print('self._yaw_lbs',type(self._yaw_lbs),self._yaw_lbs.shape)
                # Save results to self
                self._farm_power_opt_subset = farm_power_opt
                self._yaw_angles_opt_subset = yaw_angles_opt

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()
        return df_opt
